from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _fmt_marks(marks: List[int], max_len: int = 32) -> str:
    if len(marks) <= max_len:
        return ", ".join(str(v) for v in marks)
    left = ", ".join(str(v) for v in marks[: max_len // 2])
    right = ", ".join(str(v) for v in marks[-(max_len // 2) :])
    return f"{left}, ..., {right}"


def _best_by_n(summary: Dict) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for target in summary.get("targets", []):
        out[int(target["N"])] = target.get("best_overall", {})
    return out


def generate_research_report_cycle2(
    *,
    cycle1_summary_path: Path,
    cycle2_summary_path: Path,
    out_path: Path,
) -> Path:
    cycle1 = json.loads(cycle1_summary_path.read_text())
    cycle2 = json.loads(cycle2_summary_path.read_text())

    best1 = _best_by_n(cycle1)
    best2 = _best_by_n(cycle2)
    target_ns = sorted(set(best1.keys()) | set(best2.keys()))

    lines: List[str] = []
    lines.append("# Erdős #170 Cycle-2 Breakthrough Report")
    lines.append("")

    lines.append("## 1. Summary")
    lines.append("")
    overall2 = cycle2.get("overall", {})
    lines.append(f"- Cycle-2 suite: `{cycle2.get('suite', 'unknown')}`")
    lines.append(f"- Targets evaluated: {overall2.get('target_count', 0)}")
    lines.append(f"- Complete hits: {overall2.get('total_complete_hits', 0)}")
    lines.append(f"- Success targets: {overall2.get('success_targets', [])}")
    lines.append(f"- Global early-stop fired: {overall2.get('stopped_early_global', False)}")
    lines.append("")

    lines.append("## 2. Cycle-1 vs Cycle-2 Comparison")
    lines.append("")
    lines.append("| N | Cycle-1 best missing | Cycle-2 best missing | Delta | Cycle-2 stage/candidate |")
    lines.append("|---|---:|---:|---:|---|")
    for n in target_ns:
        b1 = best1.get(n, {})
        b2 = best2.get(n, {})
        m1 = int(b1.get("stage_best_missing_count", -1))
        m2 = int(b2.get("stage_best_missing_count", -1))
        delta = m2 - m1 if m1 >= 0 and m2 >= 0 else 0
        stage = b2.get("stage", "-")
        candidate = b2.get("candidate_id", "-")
        lines.append(f"| {n} | {m1} | {m2} | {delta:+d} | `{stage}` / `{candidate}` |")
    lines.append("")

    lines.append("## 3. Per-Target Outcome")
    lines.append("")
    any_complete = False
    for target in cycle2.get("targets", []):
        n = int(target["N"])
        m_try = int(target["m_try"])
        best = target["best_overall"]
        missing = best.get("stage_best_missing_list", [])

        lines.append(f"### N={n}, m_try={m_try}")
        lines.append(
            "- Best row: "
            f"`{best['stage']}` / `{best['candidate_id']}` (delete {best['deleted_mark']}), "
            f"missing={best['stage_best_missing_count']}, M25={best['stage_best_M25']}, M50={best['stage_best_M50']}."
        )
        lines.append(
            "- Move metrics: "
            f"coupled={best.get('coupled_accepts', 0)}/{best.get('coupled_attempts', 0)}, "
            f"endpoint_window={best.get('endpoint_window_accepts', 0)}/{best.get('endpoint_window_attempts', 0)}, "
            f"tail_trigger_count={best.get('tail_trigger_count', 0)}, "
            f"endpoint_hole_mode_fired={best.get('endpoint_hole_mode_fired', False)}."
        )

        if target.get("complete_target"):
            any_complete = True
            lines.append("- Status: complete `m-1` witness found and independently verified.")
            lines.append(f"- Witness marks ({len(best['stage_best_marks'])}): {_fmt_marks(best['stage_best_marks'])}")
        else:
            taxonomy = target.get("best_hole_taxonomy", {})
            lines.append("- Status: no complete witness in current Cycle-2 budget.")
            lines.append(f"- Best missing list: {missing}")
            lines.append(
                "- Hole taxonomy: "
                f"endpoint={taxonomy.get('endpoint', 0)}, "
                f"mid_scaffold={taxonomy.get('mid_scaffold', 0)}, "
                f"other={taxonomy.get('other', 0)}."
            )
        lines.append("")

    lines.append("## 4. Conclusion")
    lines.append("")
    if any_complete:
        lines.append("- Tier A achieved: at least one target reached complete `m-1` with deterministic rerun verification.")
    else:
        lines.append("- Tier B tracking: no complete witness yet; comparison table and taxonomy identify best next deletions/holes.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
