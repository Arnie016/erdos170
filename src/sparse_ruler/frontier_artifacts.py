from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FrontierRow:
    N: int
    m_try: int
    stage: str
    candidate_id: str
    deleted_mark: int
    stage_best_objective_key: Tuple[int, ...]
    stage_best_missing_count: int
    stage_best_secondary: int
    stage_best_M25: int
    stage_best_M50: int
    stage_best_marks: List[int]
    stage_best_missing_list: List[int]
    stage_best_complete_independent: bool
    complete_hits: int
    coupled_attempts: int
    coupled_accepts: int
    endpoint_window_attempts: int
    endpoint_window_accepts: int
    tail_trigger_count: int
    endpoint_hole_mode_fired: bool
    rerun_verified: Optional[bool]
    artifact_dir: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _candidate_sort_key(row: Dict[str, Any]) -> Tuple[int, ...]:
    key = tuple(int(v) for v in row.get("stage_best_objective_key", []))
    return (*key, int(row.get("deleted_mark", 0)))


def _classify_missing(missing: Iterable[int], N: int) -> Dict[str, int]:
    counts = {"endpoint": 0, "mid_scaffold": 0, "other": 0}
    lo_mid = int(0.45 * N)
    hi_mid = int(0.75 * N)
    for d in missing:
        if d <= 2 or d >= N - 2:
            counts["endpoint"] += 1
        elif lo_mid <= d <= hi_mid:
            counts["mid_scaffold"] += 1
        else:
            counts["other"] += 1
    return counts


def _fmt_marks(marks: List[int], max_len: int = 28) -> str:
    if len(marks) <= max_len:
        return ", ".join(str(v) for v in marks)
    head = ", ".join(str(v) for v in marks[: max_len // 2])
    tail = ", ".join(str(v) for v in marks[-(max_len // 2) :])
    return f"{head}, ..., {tail}"


def scan_frontier_root(root_dir: Path) -> Dict[str, Any]:
    root_dir = root_dir.resolve()
    summary_path = root_dir / "summary.json"
    summary_exists = summary_path.exists()

    candidate_files = sorted(root_dir.rglob("candidate_summary.json"))
    target_files = sorted(root_dir.rglob("target_summary.json"))
    best_files = sorted(root_dir.rglob("best.json"))

    counts_by_target_stage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for best_file in best_files:
        rel_parts = best_file.relative_to(root_dir).parts
        if len(rel_parts) >= 3:
            target_dir = rel_parts[0]
            stage_dir = rel_parts[1]
            counts_by_target_stage[target_dir][stage_dir] += 1

    best_by_n: Dict[int, Dict[str, Any]] = {}
    target_rows: List[Dict[str, Any]] = []

    if summary_exists:
        summary = _load_json(summary_path)
        for target in summary.get("targets", []):
            best = target.get("best_overall", {})
            n = int(target.get("N"))
            row = {
                "N": n,
                "m_try": int(target.get("m_try", 0)),
                "stage": best.get("stage"),
                "candidate_id": best.get("candidate_id"),
                "deleted_mark": int(best.get("deleted_mark", 0)),
                "stage_best_objective_key": tuple(int(v) for v in best.get("stage_best_objective_key", [])),
                "stage_best_missing_count": int(best.get("stage_best_missing_count", 0)),
                "stage_best_secondary": int(best.get("stage_best_secondary", 0)),
                "stage_best_M25": int(best.get("stage_best_M25", 0)),
                "stage_best_M50": int(best.get("stage_best_M50", 0)),
                "stage_best_marks": list(best.get("stage_best_marks", [])),
                "stage_best_missing_list": list(best.get("stage_best_missing_list", [])),
                "stage_best_complete_independent": bool(best.get("stage_best_complete_independent", False)),
                "complete_hits": int(best.get("complete_hits", 0)),
                "coupled_attempts": int(best.get("coupled_attempts", 0)),
                "coupled_accepts": int(best.get("coupled_accepts", 0)),
                "endpoint_window_attempts": int(best.get("endpoint_window_attempts", 0)),
                "endpoint_window_accepts": int(best.get("endpoint_window_accepts", 0)),
                "tail_trigger_count": int(best.get("tail_trigger_count", 0)),
                "endpoint_hole_mode_fired": bool(best.get("endpoint_hole_mode_fired", False)),
                "rerun_verified": best.get("rerun_verified"),
                "artifact_dir": best.get("artifact_dir", ""),
            }
            best_by_n[n] = row
            target_rows.append(row)
    else:
        for candidate_file in candidate_files:
            try:
                row = _load_json(candidate_file)
            except Exception:
                continue
            n = row.get("N")
            if n is None:
                continue
            n = int(n)
            current = best_by_n.get(n)
            if current is None or _candidate_sort_key(row) < _candidate_sort_key(current):
                best_by_n[n] = {
                    "N": n,
                    "m_try": int(row.get("m_try", 0)),
                    "stage": row.get("stage"),
                    "candidate_id": row.get("candidate_id"),
                    "deleted_mark": int(row.get("deleted_mark", 0)),
                    "stage_best_objective_key": tuple(
                        int(v) for v in row.get("stage_best_objective_key", [])
                    ),
                    "stage_best_missing_count": int(row.get("stage_best_missing_count", 0)),
                    "stage_best_secondary": int(row.get("stage_best_secondary", 0)),
                    "stage_best_M25": int(row.get("stage_best_M25", 0)),
                    "stage_best_M50": int(row.get("stage_best_M50", 0)),
                    "stage_best_marks": list(row.get("stage_best_marks", [])),
                    "stage_best_missing_list": list(row.get("stage_best_missing_list", [])),
                    "stage_best_complete_independent": bool(
                        row.get("stage_best_complete_independent", False)
                    ),
                    "complete_hits": int(row.get("complete_hits", 0)),
                    "coupled_attempts": int(row.get("coupled_attempts", 0)),
                    "coupled_accepts": int(row.get("coupled_accepts", 0)),
                    "endpoint_window_attempts": int(row.get("endpoint_window_attempts", 0)),
                    "endpoint_window_accepts": int(row.get("endpoint_window_accepts", 0)),
                    "tail_trigger_count": int(row.get("tail_trigger_count", 0)),
                    "endpoint_hole_mode_fired": bool(row.get("endpoint_hole_mode_fired", False)),
                    "rerun_verified": row.get("rerun_verified"),
                    "artifact_dir": row.get("artifact_dir", ""),
                }
        target_rows = [best_by_n[n] for n in sorted(best_by_n)]

    return {
        "root_dir": str(root_dir),
        "summary_exists": summary_exists,
        "summary_path": str(summary_path),
        "candidate_file_count": len(candidate_files),
        "target_file_count": len(target_files),
        "best_file_count": len(best_files),
        "target_rows": target_rows,
        "best_by_n": best_by_n,
        "counts_by_target_stage": {
            target: dict(stage_counts)
            for target, stage_counts in counts_by_target_stage.items()
        },
        "generated_at_utc": _utc_now_iso(),
    }


def render_exact_frontier_summary(snapshot: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Exact Frontier Summary")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{snapshot['generated_at_utc']}`")
    lines.append(f"- Frontier root: `{snapshot['root_dir']}`")
    lines.append(f"- Final summary present: `{snapshot['summary_exists']}`")
    lines.append(
        f"- Artifact counts: `best.json`={snapshot['best_file_count']}, "
        f"`candidate_summary.json`={snapshot['candidate_file_count']}, "
        f"`target_summary.json`={snapshot['target_file_count']}"
    )
    lines.append("")

    lines.append("## Current Frontier")
    lines.append("")
    if not snapshot["best_by_n"]:
        lines.append("- No candidate artifacts found yet.")
        return "\n".join(lines) + "\n"

    lines.append("| N | m_try | best stage | candidate | deleted | missing | M25 | M50 |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|")
    for n in sorted(snapshot["best_by_n"]):
        row = snapshot["best_by_n"][n]
        lines.append(
            f"| {n} | {row['m_try']} | `{row['stage']}` | `{row['candidate_id']}` | "
            f"{row['deleted_mark']} | {row['stage_best_missing_count']} | "
            f"{row['stage_best_M25']} | {row['stage_best_M50']} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    for n in sorted(snapshot["best_by_n"]):
        row = snapshot["best_by_n"][n]
        taxonomy = _classify_missing(row["stage_best_missing_list"], n)
        lines.append(
            f"- N={n}: missing={row['stage_best_missing_list']}, "
            f"taxonomy(endpoint={taxonomy['endpoint']}, mid_scaffold={taxonomy['mid_scaffold']}, "
            f"other={taxonomy['other']})."
        )

    return "\n".join(lines) + "\n"


def render_rigidity_memo(snapshot: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Rigidity Memo")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{snapshot['generated_at_utc']}`")
    lines.append(f"- Root: `{snapshot['root_dir']}`")
    lines.append("")

    lines.append("## What The Current Data Says")
    lines.append("")
    if not snapshot["best_by_n"]:
        lines.append("- No frontier artifacts are available yet.")
        return "\n".join(lines) + "\n"

    for n in sorted(snapshot["best_by_n"]):
        row = snapshot["best_by_n"][n]
        missing = row["stage_best_missing_list"]
        taxonomy = _classify_missing(missing, n)
        lines.append(f"- N={n}, m_try={row['m_try']}:")
        lines.append(
            f"  - best row: `{row['stage']}` / `{row['candidate_id']}` (delete {row['deleted_mark']})"
        )
        lines.append(
            f"  - missing={missing}; M25={row['stage_best_M25']}; M50={row['stage_best_M50']}"
        )
        lines.append(
            f"  - hole taxonomy: endpoint={taxonomy['endpoint']}, "
            f"mid_scaffold={taxonomy['mid_scaffold']}, other={taxonomy['other']}"
        )
        lines.append(
            f"  - move metrics: coupled={row['coupled_accepts']}/{row['coupled_attempts']}, "
            f"endpoint_window={row['endpoint_window_accepts']}/{row['endpoint_window_attempts']}, "
            f"tail_triggers={row['tail_trigger_count']}, endpoint_injection={row['endpoint_hole_mode_fired']}"
        )
    lines.append("")

    lines.append("## Rigidity Read")
    lines.append("")
    if len(snapshot["best_by_n"]) == 1:
        row = next(iter(snapshot["best_by_n"].values()))
        lines.append(
            "- The search is currently concentrated on a single frontier instance, so the main risk is "
            "overfitting one move class rather than expanding the evidence base."
        )
        if row["stage_best_missing_count"] > 0:
            lines.append(
                "- The best next step is to shard the same frontier across disjoint move ablations and seed bases, "
                "while keeping the current target live as the mainline run."
            )
        else:
            lines.append(
                "- A certificate is present; the next step is to freeze and publish the witness, then move the same "
                "strategy to the next N."
            )
    else:
        lines.append(
            "- Multiple N values are present. The repeated endpoint-heavy missing patterns suggest we should keep "
            "endpoint-window and coupled-move ablations in the loop, because that is where the current rigidity lives."
        )
        lines.append(
            "- For faster evidence, keep one mainline frontier run and split the rest of the compute into disjoint "
            "seed shards and move-set toggles."
        )

    lines.append("")
    lines.append("## Recommended Next Step")
    lines.append("")
    lines.append(
        "- Refresh this memo after any new candidate summary, then publish the deltas to GitHub."
    )
    lines.append(
        "- If the frontier is unchanged, do not rewrite the memo; if it changed, keep only the smallest changed "
        "artifact set plus the generated summary JSON."
    )

    return "\n".join(lines) + "\n"


def write_text_if_changed(path: Path, content: str) -> bool:
    existing = path.read_text() if path.exists() else None
    if existing == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True

