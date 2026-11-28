"""Replay and rejection controls for the frozen scoring rules."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
import json

from scoreboard import BookError, evidence_path, new_book, replay, unique_keys, update_html


def desk(identifier):
    return dict(id=identifier, label=identifier.title(), problem="A problem",
                target="A frozen target", extension="A frozen extension", prior="Prior evidence")


def event(identifier, kind, actor, **fields):
    return dict(id=identifier, type=kind, actor=actor, title=identifier,
                summary="An inspectable decision or submission.", evidence=[], **fields)


def decision(identifier, desk_id, reply_to, *, base=5, extension=False, lean=1, independent=True, **fields):
    result = event(identifier, "decision", "referee", desk=desk_id, reply_to=reply_to,
                   grade=dict(base=base, extension=extension, lean=lean, independent=independent), **fields)
    for kind, award in result["grade"].items():
        if award:
            result["evidence"].append(dict(kind=kind, label=kind, href="PROJECT_BRIEF.md",
                                           domain="Frozen test scope", method="Test receipt"))
    return result


def initial_book():
    return dict(version=1, title="Replay test", baseline="a" * 40, desks=[desk("ruler"), desk("group")],
                events=[event("ruler-opening", "submission", "ruler"),
                        event("group-opening", "submission", "group")])


class ReplayTests(TestCase):
    def test_fresh_book_uses_its_own_desks(self):
        book = new_book("c" * 40)
        self.assertEqual(replay(book)["stages"][0]["status"], "unscored")
        book["desks"][0]["id"] = "geometry"
        book["desks"][1]["id"] = "dynamics"
        book["events"] += [
            event("geometry-opening", "submission", "geometry"),
            decision("geometry-decision", "geometry", "geometry-opening", base=8, lean=0),
        ]
        self.assertEqual(replay(book)["stages"][-1]["leaders"], ["geometry"])

    def test_ties_leader_and_replacement(self):
        book = initial_book()
        book["events"] += [
            event("ruler-challenge", "challenge", "referee", desk="ruler", reply_to="ruler-opening"),
            event("ruler-response", "submission", "ruler", reply_to="ruler-challenge"),
            decision("ruler-decision", "ruler", "ruler-response"),
            decision("group-decision", "group", "group-opening"),
            event("ruler-revision", "submission", "ruler", reply_to="ruler-decision"),
            decision("ruler-revised", "ruler", "ruler-revision", base=8, extension=True,
                     lean=2, supersedes="ruler-decision"),
        ]
        result = replay(book, require_files=True)
        stages = result["stages"]
        self.assertEqual(stages[0]["status"], "unscored")
        self.assertEqual(stages[5]["leaders"], ["ruler"])
        self.assertEqual(stages[6]["status"], "tie")
        self.assertEqual(stages[7]["scores"]["ruler"]["total"], 7)
        self.assertEqual(stages[8]["scores"]["ruler"]["total"], 13)
        self.assertEqual(stages[8]["scores"]["group"]["total"], 7)
        self.assertEqual(stages[8]["status"], "leader")

    def test_only_referee_awards_and_links_stay_on_desk(self):
        book = initial_book()
        forged = decision("ruler-decision", "ruler", "ruler-opening")
        forged["actor"] = "ruler"
        book["events"].append(forged)
        with self.assertRaisesRegex(BookError, "Only the referee"):
            replay(book)
        forged["actor"] = "referee"
        forged["reply_to"] = "group-opening"
        with self.assertRaisesRegex(BookError, "latest submission"):
            replay(book)

    def test_award_bounds_and_evidence(self):
        book = initial_book()
        valid = decision("ruler-decision", "ruler", "ruler-opening")
        for key, value in (("base", True), ("base", 7), ("extension", True), ("lean", 2), ("lean", 4)):
            with self.subTest(key=key, value=value):
                malformed = deepcopy(valid)
                malformed["grade"][key] = value
                book["events"] = book["events"][:2] + [malformed]
                with self.assertRaises(BookError):
                    replay(book)
        valid["evidence"] = valid["evidence"][:1]
        book["events"] = book["events"][:2] + [valid]
        with self.assertRaisesRegex(BookError, "lacks evidence"):
            replay(book)

    def test_revised_award_must_name_the_award_it_replaces(self):
        book = initial_book()
        book["events"] += [
            decision("ruler-decision", "ruler", "ruler-opening"),
            event("ruler-appeal", "submission", "ruler", reply_to="ruler-decision"),
            decision("ruler-revised", "ruler", "ruler-appeal"),
        ]
        with self.assertRaisesRegex(BookError, "identify the award"):
            replay(book)

    def test_unsafe_links_and_duplicate_keys_are_rejected(self):
        for href in ("javascript:alert(1)", "//example.org/file", "../../../../etc/passwd",
                     "%2Fetc%2Fpasswd", "https://github.com/owner/repo/blob/main/file"):
            with self.subTest(href=href), self.assertRaises(BookError):
                evidence_path(href)
        with self.assertRaisesRegex(BookError, "Duplicate JSON key"):
            json.loads('{"version":1,"version":2}', object_pairs_hook=unique_keys)

    def test_embedded_replay_is_deterministic_and_escapes_html(self):
        book = initial_book()
        book["title"] = "</script><script>alert(1)</script>"
        result = replay(book)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "index.html"
            path.write_text('<script id="match-data" type="application/json">\n{}\n</script>', encoding="utf-8")
            with self.assertRaisesRegex(BookError, "stale"):
                update_html(path, result, write=False)
            update_html(path, result, write=True)
            update_html(path, result, write=False)
            self.assertNotIn("<script>alert", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
