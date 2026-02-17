from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _classify_missing_list(missing: List[int], N: int) -> Dict[str, int]:
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


def _format_marks(marks: List[int], max_len: int = 30) -> str:
    if len(marks) <= max_len:
        return ", ".join(str(v) for v in marks)
    head = ", ".join(str(v) for v in marks[: max_len // 2])
    tail = ", ".join(str(v) for v in marks[-(max_len // 2) :])
    return f"{head}, ..., {tail}"


def generate_research_report(
    *,
    baseline_summary_path: Path,
    e_hunt_summary_path: Path,
    out_path: Path,
) -> Path:
    baseline = json.loads(baseline_summary_path.read_text())
    e_hunt = json.loads(e_hunt_summary_path.read_text())

    lines: List[str] = []
    lines.append("# Erd\u0151s #170 Cycle 1 Research Report")
    lines.append("")

    lines.append("## 1. Definitions")
    lines.append("")
    lines.append("- `best_missing`: minimum uncovered-distance count found across attempts for a fixed `(N,m)` setting.")
    lines.append("- `complete`: all distances `1..N` are represented, so `missing_count = 0`.")
    lines.append("- `excess E`: `E = m - nint(sqrt(3N + 9/4))` where `nint` is round-half-up.")
    lines.append("")

    lines.append("## 2. Deterministic Baseline Findings")
    lines.append("")
    ratio = baseline.get("ratio", {})
    lines.append(
        "- Sweep range: "
        f"N={baseline['range']['start']}..{baseline['range']['end']} step={baseline['range']['step']} "
        f"({baseline['range']['count']} points)."
    )
    lines.append(
        f"- Ratio envelope m/sqrt(N): min={ratio.get('min', 0):.6f}, max={ratio.get('max', 0):.6f}, mean={ratio.get('mean', 0):.6f}."
    )
    lines.append(
        f"- Completeness failures in deterministic constructor: {len(baseline.get('complete_failures', []))}."
    )

    fixed_points = baseline.get("fixed_points", [])
    if fixed_points:
        lines.append("- Fixed points:")
        for row in fixed_points:
            lines.append(
                f"  - N={row['N']}: m={row['m']}, base_term={row['base_term']}, E={row['E']}, "
                f"m/sqrt(N)={row['m_sqrtN']:.6f}, source={row['source']}, missing={row['missing_count']}"
            )
    lines.append("")

    lines.append("## 3. E-Hunt Outcomes by N")
    lines.append("")

    any_success = False
    for target in e_hunt.get("targets", []):
        N = int(target["N"])
        m_try = int(target["m_try"])
        best = target["best_overall"]
        missing_list = list(best.get("stage_best_missing_list", []))
        taxonomy = target.get("best_hole_taxonomy") or _classify_missing_list(missing_list, N)

        lines.append(f"### N={N}, m_try={m_try}")
        lines.append(
            f"- Best stage/candidate: `{best['stage']}` / `{best['candidate_id']}` (deleted mark {best['deleted_mark']})."
        )
        lines.append(
            f"- Best objective key: {best['stage_best_objective_key']}; "
            f"missing={best['stage_best_missing_count']}, M25={best['stage_best_M25']}, M50={best['stage_best_M50']}."
        )
        lines.append(
            f"- Coupled moves: attempts={best['coupled_attempts']}, accepts={best['coupled_accepts']}; "
            f"endpoint-hole fired={best['endpoint_hole_mode_fired']}."
        )

        if target.get("complete_target"):
            any_success = True
            lines.append("- Status: complete witness found and independently verified.")
            lines.append(
                f"- Witness marks ({len(best['stage_best_marks'])}): {_format_marks(best['stage_best_marks'])}"
            )
        else:
            lines.append("- Status: no complete m-1 witness in this run budget.")
            lines.append(f"- Best missing list: {missing_list}")
            lines.append(
                "- Hole taxonomy: "
                f"endpoint={taxonomy['endpoint']}, "
                f"mid_scaffold={taxonomy['mid_scaffold']}, other={taxonomy['other']}"
            )
        lines.append("")

    lines.append("## 4. Summary")
    lines.append("")
    if any_success:
        lines.append("- At least one target reached complete `m-1` with independent rerun verification.")
    else:
        lines.append(
            "- No complete `m-1` witness was found in this cycle; results remain useful as reproducible near-miss data."
        )
        lines.append("- Persistent-hole taxonomy is included above to drive next coupled-move targeting.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
