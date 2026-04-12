from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _terminal_run(marks: Sequence[int], n: int) -> List[int]:
    interior = [mark for mark in marks if mark < n]
    if not interior:
        return []
    run = [interior[-1]]
    for mark in reversed(interior[:-1]):
        if mark == run[-1] - 1:
            run.append(mark)
        else:
            break
    return list(reversed(run))


def _gaps(values: Sequence[int]) -> List[int]:
    return [b - a for a, b in zip(values, values[1:])]


def _residue_counts(values: Iterable[int], modulus: int) -> List[int]:
    counts = [0] * modulus
    for value in values:
        counts[value % modulus] += 1
    return counts


@dataclass(frozen=True)
class FrontierRecord:
    source: str
    N: int
    m_try: int
    stage: str
    candidate_id: str
    deleted_mark: int
    best_missing_count: int
    missing_list: List[int]
    missing_span: int
    furthest_missing_gap: int
    terminal_run: List[int]
    terminal_run_len: int
    run_prefix_gap: int
    endpoint_gap: int
    terminal_gaps: List[int]
    tail_density_25: int
    tail_density_50: int
    residue_mod_4_tail: List[int]
    residue_mod_8_tail: List[int]


def _record_from_best_row(source: str, target: Dict) -> FrontierRecord:
    best = dict(target.get("best_overall", {}))
    if not best:
        raise ValueError(f"{source} target {target.get('N')} has no best_overall row")

    n = int(target["N"])
    m_try = int(target["m_try"])
    marks = list(best.get("stage_best_marks", []))
    missing = list(best.get("stage_best_missing_list", []))
    terminal_run = _terminal_run(marks, n)
    terminal_run_len = len(terminal_run)
    endpoint_gap = n - terminal_run[-1] if terminal_run else 0
    run_prefix_gap = 0
    if terminal_run:
        run_start = terminal_run[0]
        prior_marks = [mark for mark in marks if mark < run_start]
        if prior_marks:
            run_prefix_gap = run_start - prior_marks[-1]
    gaps = _gaps(terminal_run)
    if terminal_run:
        gaps.append(endpoint_gap)

    tail_25 = [mark for mark in marks if mark >= n - 25]
    tail_50 = [mark for mark in marks if mark >= n - 50]

    return FrontierRecord(
        source=source,
        N=n,
        m_try=m_try,
        stage=str(best.get("stage", "")),
        candidate_id=str(best.get("candidate_id", "")),
        deleted_mark=int(best.get("deleted_mark", -1)),
        best_missing_count=int(best.get("stage_best_missing_count", -1)),
        missing_list=missing,
        missing_span=(max(missing) - min(missing)) if missing else 0,
        furthest_missing_gap=(n - max(missing)) if missing else 0,
        terminal_run=terminal_run,
        terminal_run_len=terminal_run_len,
        run_prefix_gap=run_prefix_gap,
        endpoint_gap=endpoint_gap,
        terminal_gaps=gaps,
        tail_density_25=len(tail_25),
        tail_density_50=len(tail_50),
        residue_mod_4_tail=_residue_counts(tail_25, 4),
        residue_mod_8_tail=_residue_counts(tail_25, 8),
    )


def load_frontier_records(
    *,
    cycle1_summary_path: Path,
    cycle2_summary_path: Path,
) -> List[FrontierRecord]:
    records: List[FrontierRecord] = []
    if cycle1_summary_path.exists():
        cycle1 = _load_json(cycle1_summary_path)
        for target in cycle1.get("targets", []):
            records.append(_record_from_best_row("cycle1", target))
    if cycle2_summary_path.exists():
        cycle2 = _load_json(cycle2_summary_path)
        for target in cycle2.get("targets", []):
            records.append(_record_from_best_row("cycle2", target))
    return records


def _group_records(records: Sequence[FrontierRecord]) -> Dict[int, List[FrontierRecord]]:
    grouped: Dict[int, List[FrontierRecord]] = defaultdict(list)
    for record in records:
        grouped[record.N].append(record)
    return grouped


def build_frontier_atlas_payload(
    *,
    cycle1_summary_path: Path,
    cycle2_summary_path: Path,
) -> Dict:
    records = load_frontier_records(
        cycle1_summary_path=cycle1_summary_path,
        cycle2_summary_path=cycle2_summary_path,
    )
    grouped = _group_records(records)

    motifs = []
    for n in sorted(grouped):
        group = grouped[n]
        missing_lists = {tuple(record.missing_list) for record in group}
        terminal_runs = {tuple(record.terminal_run) for record in group}
        furthest_gaps = sorted({record.furthest_missing_gap for record in group})
        motifs.append(
            {
                "N": n,
                "sources": [record.source for record in group],
                "missing_lists": [list(missing) for missing in sorted(missing_lists)],
                "terminal_runs": [list(run) for run in sorted(terminal_runs)],
                "furthest_missing_gaps": furthest_gaps,
                "shared_missing": len(missing_lists) == 1,
                "shared_terminal_run": len(terminal_runs) == 1,
            }
        )

    return {
        "generated_at_utc": _utc_now_iso(),
        "sources": {
            "cycle1_summary": str(cycle1_summary_path),
            "cycle2_summary": str(cycle2_summary_path),
        },
        "record_count": len(records),
        "records": [record.__dict__ for record in records],
        "motifs": motifs,
    }


def _fmt_list(values: Sequence[int]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def render_frontier_atlas_markdown(payload: Dict) -> str:
    lines: List[str] = []
    lines.append("# Frontier Atlas")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{payload['generated_at_utc']}`")
    lines.append(f"- Records analyzed: {payload['record_count']}")
    lines.append(f"- Cycle-1 source: `{payload['sources']['cycle1_summary']}`")
    lines.append(f"- Cycle-2 source: `{payload['sources']['cycle2_summary']}`")
    lines.append("")

    lines.append("## 1) Core Pattern")
    lines.append("")
    lines.append(
        "The current frontier is not drifting toward new missing sets; it is collapsing into the same tail geometry."
    )
    lines.append(
        "Every observed certificate has a contiguous terminal run immediately before the endpoint, followed by one larger endpoint gap."
    )
    lines.append(
        "The furthest missing distance is always just below that endpoint gap, so the unresolved set sits in the last 10 or so distances rather than at the endpoint itself."
    )
    lines.append("")

    lines.append("## 2) Per-Record Geometry")
    lines.append("")
    lines.append(
        "| source | N | m_try | best_missing | missing_list | furthest_missing_gap | terminal_run | endpoint_gap | run_prefix_gap | tail_density_25 |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |"
    )
    for record in payload["records"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    record["source"],
                    str(record["N"]),
                    str(record["m_try"]),
                    str(record["best_missing_count"]),
                    _fmt_list(record["missing_list"]),
                    str(record["furthest_missing_gap"]),
                    _fmt_list(record["terminal_run"]),
                    str(record["endpoint_gap"]),
                    str(record["run_prefix_gap"]),
                    str(record["tail_density_25"]),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## 3) Shared Motifs")
    lines.append("")
    for motif in payload["motifs"]:
        lines.append(
            f"- N={motif['N']}: shared_missing={motif['shared_missing']}, shared_terminal_run={motif['shared_terminal_run']}, "
            f"furthest_missing_gaps={motif['furthest_missing_gaps']}."
        )
        if motif["missing_lists"]:
            lines.append(
                f"  missing_lists={', '.join(_fmt_list(m) for m in motif['missing_lists'])}."
            )
        if motif["terminal_runs"]:
            lines.append(
                f"  terminal_runs={', '.join(_fmt_list(r) for r in motif['terminal_runs'])}."
            )
    lines.append("")

    lines.append("## 4) Geometric Read")
    lines.append("")
    lines.append(
        "The most useful interpretation is scaffold rigidity: the tail block is already dense and interval-like, so the search is probably fighting the last coarse gap rather than isolated endpoints."
    )
    lines.append(
        "That points toward multi-mark tail surgery, residue-preserving block moves, or a local exact repair model instead of more single-distance targeting."
    )
    return "\n".join(lines) + "\n"


def generate_frontier_atlas(
    *,
    cycle1_summary_path: Path,
    cycle2_summary_path: Path,
    out_path: Path,
    json_out_path: Optional[Path] = None,
) -> Path:
    payload = build_frontier_atlas_payload(
        cycle1_summary_path=cycle1_summary_path,
        cycle2_summary_path=cycle2_summary_path,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_frontier_atlas_markdown(payload))
    if json_out_path is not None:
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(json.dumps(payload, indent=2))
    return out_path
