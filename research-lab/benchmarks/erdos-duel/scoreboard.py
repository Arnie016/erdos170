#!/usr/bin/env python3
"""Replay the referee's recorded decisions and sync the standalone board."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import unquote, urlsplit


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
MAXIMUM = 13
BASE_LABELS = {
    0: "No advance",
    1: "Reproducible obstruction",
    3: "Exact finite result",
    5: "Proved subcase",
    8: "Frozen target settled",
}
GRADE_KEYS = {"base", "extension", "lean", "independent"}
EVIDENCE_KINDS = GRADE_KEYS | {"context"}
IDENTIFIER = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
PINNED_LINK = re.compile(
    r"(?:https://github\.com/[\w.-]+/[\w.-]+/blob/[0-9a-f]{40}/[^?#]+"
    r"|https://arxiv\.org/(?:abs|html|pdf)/\d{4}\.\d{4,5}v\d+)(?:#[\w.:-]+)?\Z"
)
HTML_DATA = re.compile(
    r'(<script id="match-data" type="application/json">\n)(.*?)(\n</script>)',
    re.DOTALL,
)


class BookError(ValueError):
    """A match book violates the replay contract."""


def unique_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise BookError(f"Duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def require_keys(value, required, optional, where):
    if not isinstance(value, dict):
        raise BookError(f"{where} must be an object")
    missing = required - value.keys()
    unexpected = value.keys() - required - optional
    if missing or unexpected:
        raise BookError(f"{where}: missing {sorted(missing)}, unexpected {sorted(unexpected)}")


def require_text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise BookError(f"{where} must be nonempty text")


def require_id(value, where):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise BookError(f"{where} must be a lowercase identifier")


def evidence_path(href):
    require_text(href, "Evidence link")
    if href != href.strip() or "\\" in href or any(ord(character) < 32 for character in href):
        raise BookError(f"Unsafe evidence link {href!r}")
    try:
        parts = urlsplit(href)
    except ValueError as error:
        raise BookError(f"Malformed evidence link {href!r}") from error
    decoded = unquote(parts.path)
    relative = PurePosixPath(decoded)
    if "\\" in decoded or any(ord(character) < 32 for character in decoded):
        raise BookError(f"Unsafe evidence path {href!r}")
    if PINNED_LINK.fullmatch(href) and ".." not in relative.parts:
        return None
    if parts.scheme or parts.netloc or parts.query or relative.is_absolute() or not decoded:
        raise BookError(f"Evidence must be a relative file or pinned source: {href!r}")
    path = (HERE / relative).resolve()
    if not path.is_relative_to(REPOSITORY):
        raise BookError(f"Evidence escapes the repository: {href!r}")
    return path


def check_evidence(evidence, require_files):
    if not isinstance(evidence, list):
        raise BookError("Evidence must be a list")
    kinds = set()
    seen = set()
    for item in evidence:
        require_keys(item, {"kind", "label", "href", "domain", "method"}, set(), "Evidence")
        for field, value in item.items():
            require_text(value, f"Evidence {field}")
        if item["kind"] not in EVIDENCE_KINDS:
            raise BookError(f"Unknown evidence kind {item['kind']!r}")
        identity = item["kind"], item["href"]
        if identity in seen:
            raise BookError(f"Repeated evidence {identity!r}")
        seen.add(identity)
        path = evidence_path(item["href"])
        if require_files and path is not None and not path.is_file():
            raise BookError(f"Missing evidence file {item['href']!r}")
        kinds.add(item["kind"])
    return kinds


def checked_grade(grade, kinds):
    require_keys(grade, GRADE_KEYS, set(), "Grade")
    if type(grade["base"]) is not int or grade["base"] not in BASE_LABELS:
        raise BookError("Base points must be 0, 1, 3, 5, or 8")
    if type(grade["lean"]) is not int or grade["lean"] not in (0, 1, 2):
        raise BookError("Lean points must be 0, 1, or 2")
    if any(type(grade[key]) is not bool for key in ("extension", "independent")):
        raise BookError("Extension and independent awards must be booleans")
    if (grade["extension"] or grade["lean"] == 2) and grade["base"] != 8:
        raise BookError("Full extension and full-target Lean require target acceptance")
    missing = {key for key, value in grade.items() if value} - kinds
    if missing:
        raise BookError(f"Award lacks evidence for {', '.join(sorted(missing))}")
    return grade


def snapshot(event_id, decisions, desks):
    scores = {}
    for desk in desks:
        decision = decisions.get(desk)
        grade = decision["grade"] if decision else dict(base=0, extension=False, lean=0, independent=False)
        points = {
            "base": grade["base"],
            "extension": 2 * grade["extension"],
            "lean": grade["lean"],
            "independent": int(grade["independent"]),
        }
        scores[desk] = {
            "total": sum(points.values()),
            "points": points,
            "base_label": BASE_LABELS[grade["base"]],
            "decision": decision["id"] if decision else None,
        }
    best = max(score["total"] for score in scores.values())
    leaders = [desk for desk, score in scores.items() if score["total"] == best]
    return {
        "event": event_id,
        "scores": scores,
        "leaders": leaders,
        "status": "unscored" if not decisions else "tie" if len(leaders) == 2 else "leader",
    }


def replay(book, *, require_files=False):
    require_keys(book, {"version", "title", "baseline", "desks", "events"}, set(), "Match")
    if type(book["version"]) is not int or book["version"] != 1:
        raise BookError("Only match schema version 1 is supported")
    require_text(book["title"], "Title")
    if not isinstance(book["baseline"], str) or not COMMIT.fullmatch(book["baseline"]):
        raise BookError("Baseline must be a full Git commit hash")
    if not isinstance(book["desks"], list) or len(book["desks"]) != 2:
        raise BookError("Exactly two desks are required")
    desks = {}
    for desk in book["desks"]:
        require_keys(desk, {"id", "label", "problem", "target", "extension", "prior"}, set(), "Desk")
        require_id(desk["id"], "Desk ID")
        if desk["id"] in desks or desk["id"] == "referee":
            raise BookError(f"Duplicate or reserved desk ID {desk['id']!r}")
        for key in ("label", "problem", "target", "extension", "prior"):
            require_text(desk[key], f"Desk {key}")
        desks[desk["id"]] = desk
    if not isinstance(book["events"], list):
        raise BookError("Events must be an ordered list")

    ids = set()
    last = {}
    submissions = {}
    decisions = {}
    stages = [snapshot(None, decisions, desks)]
    common = {"id", "type", "actor", "title", "summary", "evidence"}
    for event in book["events"]:
        if not isinstance(event, dict):
            raise BookError("Event must be an object")
        kind = event.get("type")
        actor = event.get("actor")
        if kind == "submission":
            require_keys(event, common, {"reply_to"}, "Submission")
            if not isinstance(actor, str) or actor not in desks:
                raise BookError("Only a desk can submit")
            desk = actor
            previous = last.get(desk)
            if previous is None and "reply_to" in event:
                raise BookError("An opening submission has no parent event")
            if previous is not None and (
                previous["type"] not in ("challenge", "decision")
                or event.get("reply_to") != previous["id"]
            ):
                raise BookError("A follow-up submission must answer the desk's latest referee event")
        elif kind in ("challenge", "decision"):
            required = common | {"desk", "reply_to"}
            if kind == "decision":
                required |= {"grade"}
            require_keys(event, required, {"supersedes"} if kind == "decision" else set(), "Referee event")
            if actor != "referee":
                raise BookError("Only the referee can challenge or award points")
            desk = event["desk"]
            if not isinstance(desk, str) or desk not in desks:
                raise BookError("Referee event must name a desk")
            previous = last.get(desk)
            if previous is None or previous["type"] not in ("submission", "challenge"):
                raise BookError("Referee event needs a current submission")
            if kind == "challenge" and previous["type"] != "submission":
                raise BookError("A challenge must follow a submission")
            if event["reply_to"] != submissions[desk]["id"]:
                raise BookError("Referee event must address the desk's latest submission")
        else:
            raise BookError(f"Unknown event type {kind!r}")

        event_id = event["id"]
        require_id(event_id, "Event ID")
        if event_id in ids or event_id == "baseline":
            raise BookError(f"Duplicate or reserved event ID {event_id!r}")
        ids.add(event_id)
        require_text(event["title"], "Event title")
        require_text(event["summary"], "Event summary")
        kinds = check_evidence(event["evidence"], require_files)
        if kind == "submission":
            submissions[desk] = event
        if kind == "decision":
            old = decisions.get(desk)
            if old is None and "supersedes" in event:
                raise BookError("An initial decision cannot replace an earlier award")
            if old is not None and event.get("supersedes") != old["id"]:
                raise BookError("A revised decision must identify the award it replaces")
            checked_grade(event["grade"], kinds)
            decisions[desk] = event
        last[desk] = event
        stages.append(snapshot(event_id, decisions, desks))
    return {"maximum": MAXIMUM, "book": book, "stages": stages}


def update_html(path, replayed, *, write):
    template = path.read_text(encoding="utf-8")
    encoded = json.dumps(replayed, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    compiled, count = HTML_DATA.subn(lambda match: match[1] + encoded + match[3], template)
    if count != 1:
        raise BookError("The board must contain exactly one match-data script")
    if write:
        path.write_text(compiled, encoding="utf-8")
    elif compiled != template:
        raise BookError("Embedded replay is stale. Run scoreboard.py --write-html.")


def new_book(baseline):
    book = {
        "version": 1,
        "title": "Proof duel",
        "baseline": baseline,
        "desks": [
            {
                "id": f"desk-{number}",
                "label": f"Desk {number}",
                "problem": "Name the problem in the frozen brief.",
                "target": "State the exact target before play.",
                "extension": "State the exact extension before play.",
                "prior": "Record what has already been proved before play.",
            }
            for number in (1, 2)
        ],
        "events": [],
    }
    replay(book)
    return book


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, default=HERE / "match.json")
    parser.add_argument("--template", metavar="COMMIT", help="print a new two-desk match book")
    parser.add_argument("--check-evidence", action="store_true", help="require every relative receipt to exist")
    parser.add_argument("--stage", help="show one event by ID, or baseline")
    parser.add_argument("--json", action="store_true", help="emit the book and every computed stage")
    html = parser.add_mutually_exclusive_group()
    html.add_argument("--write-html", action="store_true", help="sync the executable board")
    html.add_argument("--check-html", action="store_true", help="fail if the board data is stale")
    args = parser.parse_args()
    try:
        if args.template is not None:
            if args.stage or args.write_html or args.check_html:
                parser.error("--template prints a fresh book. Validate or sync it in a separate call.")
            print(json.dumps(new_book(args.template), ensure_ascii=False, indent=2))
            return
        book = json.loads(args.book.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
        result = replay(book, require_files=args.check_evidence)
        stages = {stage["event"] or "baseline": stage for stage in result["stages"]}
        if args.stage and args.stage not in stages:
            raise BookError(f"Unknown stage {args.stage!r}")
        if args.write_html or args.check_html:
            update_html(HERE / "index.html", result, write=args.write_html)
        if args.json:
            data = stages[args.stage] if args.stage else result
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return
        stage = stages[args.stage] if args.stage else result["stages"][-1]
        print(book["title"])
        print(f"Stage {stage['event'] or 'baseline'} · {len(book['events'])} recorded events")
        for desk in book["desks"]:
            score = stage["scores"][desk["id"]]
            label = score["base_label"] if score["decision"] else "Unscored"
            print(f"{desk['label']} {score['total']}/{MAXIMUM} · {label}")
        leaders = [desk["label"] for desk in book["desks"] if desk["id"] in stage["leaders"]]
        print("Awaiting a referee decision" if stage["status"] == "unscored" else
              f"Tie between {' and '.join(leaders)}" if stage["status"] == "tie" else
              f"Leader {leaders[0]}")
    except (OSError, ValueError) as error:
        print(f"Cannot replay match. {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
