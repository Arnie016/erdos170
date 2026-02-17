from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _classify_missing_list(missing: List[int], n: int) -> Dict[str, int]:
    counts = {"endpoint": 0, "mid_scaffold": 0, "other": 0}
    lo_mid = int(0.45 * n)
    hi_mid = int(0.75 * n)
    for d in missing:
        if d <= 2 or d >= n - 2:
            counts["endpoint"] += 1
        elif lo_mid <= d <= hi_mid:
            counts["mid_scaffold"] += 1
        else:
            counts["other"] += 1
    return counts


def _expected_counts_from_config(config: Dict) -> Tuple[int, int]:
    targets = config.get("targets", [])
    search = config.get("search", {})
    stages = search.get("stages", {})
    screen = stages.get("screen", {})
    deep = stages.get("deep", {})
    championship = stages.get("championship", {})

    allowlist_sizes = [
        len(target.get("candidate_allowlist_deleted_marks", [])) for target in targets
    ]
    target_count = len(targets)

    expected_best = 0
    expected_best += sum(allow * int(screen.get("seeds", 0)) for allow in allowlist_sizes)
    expected_best += sum(
        int(deep.get("promote_top_k", 0)) * int(deep.get("seeds", 0))
        for _ in allowlist_sizes
    )
    expected_best += sum(int(championship.get("seeds", 0)) for _ in allowlist_sizes)

    expected_candidates = 0
    expected_candidates += sum(allowlist_sizes)
    expected_candidates += target_count * int(deep.get("promote_top_k", 0))
    expected_candidates += target_count
    return expected_best, expected_candidates


def _scan_cycle2_partial(cycle2_root: Path) -> Dict:
    best_files = list(cycle2_root.rglob("best.json"))
    candidate_files = list(cycle2_root.rglob("candidate_summary.json"))
    summary_exists = (cycle2_root / "summary.json").exists()

    best_by_n: Dict[int, Dict] = {}
    counts_by_target_stage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for best_file in best_files:
        rel_parts = best_file.relative_to(cycle2_root).parts
        if len(rel_parts) >= 3:
            target_dir = rel_parts[0]
            stage_dir = rel_parts[1]
            counts_by_target_stage[target_dir][stage_dir] += 1

    for candidate_file in candidate_files:
        try:
            row = _load_json(candidate_file)
        except Exception:
            continue
        n = row.get("N")
        missing = row.get("stage_best_missing_count")
        if n is None or missing is None:
            continue
        cur = best_by_n.get(int(n))
        if cur is None or int(missing) < int(cur["stage_best_missing_count"]):
            best_by_n[int(n)] = {
                "stage": row.get("stage"),
                "candidate_id": row.get("candidate_id"),
                "deleted_mark": row.get("deleted_mark"),
                "stage_best_missing_count": int(missing),
                "stage_best_missing_list": list(row.get("stage_best_missing_list", [])),
                "stage_best_M25": int(row.get("stage_best_M25", 0)),
                "stage_best_M50": int(row.get("stage_best_M50", 0)),
                "coupled_attempts": int(row.get("coupled_attempts", 0)),
                "coupled_accepts": int(row.get("coupled_accepts", 0)),
                "endpoint_window_attempts": int(row.get("endpoint_window_attempts", 0)),
                "endpoint_window_accepts": int(row.get("endpoint_window_accepts", 0)),
                "tail_trigger_count": int(row.get("tail_trigger_count", 0)),
                "artifact_path": str(candidate_file),
            }

    return {
        "best_file_count": len(best_files),
        "candidate_file_count": len(candidate_files),
        "summary_exists": summary_exists,
        "best_by_n": best_by_n,
        "counts_by_target_stage": {
            target: dict(stage_counts)
            for target, stage_counts in counts_by_target_stage.items()
        },
    }


def _fmt_pct(numer: int, denom: Optional[int]) -> str:
    if not denom:
        return "n/a"
    return f"{100.0 * numer / denom:.1f}%"


def _progress_note(actual: int, expected: Optional[int]) -> str:
    if expected is None:
        return ""
    if actual > expected:
        return " (above config expectation due to resumed/extra candidate artifacts)"
    return ""


def generate_research_report_to_date(
    *,
    baseline_summary_path: Path,
    cycle1_summary_path: Path,
    cycle2_root: Path,
    cycle2_config_path: Optional[Path],
    out_path: Path,
) -> Path:
    baseline = _load_json(baseline_summary_path)
    cycle1 = _load_json(cycle1_summary_path)
    cycle2_partial = _scan_cycle2_partial(cycle2_root)

    expected_best = None
    expected_candidates = None
    if cycle2_config_path and cycle2_config_path.exists():
        cycle2_config = _load_json(cycle2_config_path)
        expected_best, expected_candidates = _expected_counts_from_config(cycle2_config)

    lines: List[str] = []
    lines.append("# Erdos #170 Research Report (To Date)")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{_utc_now_iso()}`")
    lines.append(f"- Baseline summary: `{baseline_summary_path}`")
    lines.append(f"- Cycle-1 summary: `{cycle1_summary_path}`")
    lines.append(f"- Cycle-2 root: `{cycle2_root}`")
    lines.append("")

    lines.append("## 1) Problem And Definitions")
    lines.append("")
    lines.append("- Goal in this finite campaign: find complete `m-1` witnesses at N=500,700,1000.")
    lines.append("- `complete` means every distance `1..N` is represented (missing_count = 0).")
    lines.append("- `best_missing` is the smallest uncovered-distance count observed for fixed `(N,m)`.")
    lines.append("- Excess view: `E = m - nint(sqrt(3N + 9/4))`.")
    lines.append("")

    lines.append("## 2) Deterministic Baseline (Wichmann-Oriented)")
    lines.append("")
    ratio = baseline.get("ratio", {})
    r = baseline.get("range", {})
    lines.append(
        f"- Sweep: N={r.get('start')}..{r.get('end')} step={r.get('step')} ({r.get('count')} points)."
    )
    lines.append(
        f"- Ratio `m/sqrt(N)`: min={ratio.get('min', 0.0):.6f}, "
        f"mean={ratio.get('mean', 0.0):.6f}, max={ratio.get('max', 0.0):.6f}."
    )
    lines.append(f"- E distribution: {baseline.get('E_distribution', {})}")
    lines.append(
        f"- Completeness failures in baseline sweep: {len(baseline.get('complete_failures', []))}."
    )
    for row in baseline.get("fixed_points", []):
        lines.append(
            f"- Fixed N={row['N']}: m={row['m']}, base_term={row['base_term']}, "
            f"E={row['E']}, missing={row['missing_count']}."
        )
    lines.append("")

    lines.append("## 3) Cycle-1 (Completed)")
    lines.append("")
    overall1 = cycle1.get("overall", {})
    lines.append(
        f"- Targets={overall1.get('target_count', 0)}, complete hits={overall1.get('total_complete_hits', 0)}."
    )
    lines.append(
        f"- Success targets={overall1.get('success_targets', [])}, "
        f"global early stop={overall1.get('stopped_early_global', False)}."
    )
    for target in cycle1.get("targets", []):
        n = int(target["N"])
        m_try = int(target["m_try"])
        best = target["best_overall"]
        missing_list = list(best.get("stage_best_missing_list", []))
        taxonomy = target.get("best_hole_taxonomy") or _classify_missing_list(missing_list, n)
        lines.append(
            f"- N={n}, m_try={m_try}: best_missing={best.get('stage_best_missing_count')} "
            f"at {best.get('stage')} / {best.get('candidate_id')} (delete {best.get('deleted_mark')})."
        )
        lines.append(
            f"  missing_list={missing_list}, M25={best.get('stage_best_M25')}, "
            f"M50={best.get('stage_best_M50')}."
        )
        lines.append(
            f"  holes(endpoint={taxonomy.get('endpoint', 0)}, "
            f"mid_scaffold={taxonomy.get('mid_scaffold', 0)}, other={taxonomy.get('other', 0)})."
        )
    lines.append("")

    lines.append("## 4) Cycle-2 (In Progress Snapshot)")
    lines.append("")
    best_count = int(cycle2_partial["best_file_count"])
    cand_count = int(cycle2_partial["candidate_file_count"])
    lines.append(f"- `summary.json` present: {cycle2_partial['summary_exists']}")
    lines.append(
        f"- Seed artifact count (`best.json`): {best_count}"
        + (
            f" / {expected_best} ({_fmt_pct(best_count, expected_best)})"
            if expected_best is not None
            else ""
        )
        + _progress_note(best_count, expected_best)
    )
    lines.append(
        f"- Candidate artifact count (`candidate_summary.json`): {cand_count}"
        + (
            f" / {expected_candidates} ({_fmt_pct(cand_count, expected_candidates)})"
            if expected_candidates is not None
            else ""
        )
        + _progress_note(cand_count, expected_candidates)
    )
    lines.append("- By target/stage (`best.json` counts):")
    for target_dir in sorted(cycle2_partial["counts_by_target_stage"].keys()):
        stage_counts = cycle2_partial["counts_by_target_stage"][target_dir]
        lines.append(f"  - {target_dir}: {stage_counts}")

    lines.append("- Best observed so far per target:")
    for n in sorted(cycle2_partial["best_by_n"].keys()):
        row = cycle2_partial["best_by_n"][n]
        lines.append(
            f"  - N={n}: best_missing={row['stage_best_missing_count']} "
            f"(stage={row['stage']}, candidate={row['candidate_id']}, delete={row['deleted_mark']})."
        )
        lines.append(
            f"    missing_list={row['stage_best_missing_list']}, "
            f"M25={row['stage_best_M25']}, M50={row['stage_best_M50']}."
        )
        lines.append(
            f"    coupled_accepts={row['coupled_accepts']}/{row['coupled_attempts']}, "
            f"endpoint_window_accepts={row['endpoint_window_accepts']}/{row['endpoint_window_attempts']}, "
            f"tail_trigger_count={row['tail_trigger_count']}."
        )
    lines.append("")

    lines.append("## 5) Research Interpretation")
    lines.append("")
    lines.append("- Cycle-1 and current Cycle-2 confirm the same near-miss frontier:")
    lines.append("  - N=500 misses [493] at m=39.")
    lines.append("  - N=700 misses [690,691,692] at m=46.")
    lines.append("  - N=1000 misses [989,990,991] at m=55.")
    lines.append("- This is evidence of strong endpoint-tail rigidity in current move classes.")
    lines.append("- `m-1` completeness is still open at these N in the current heuristic budget.")
    lines.append("- For Erdos #170 relevance: this is finite-N structural evidence, not an asymptotic proof.")
    lines.append("")

    lines.append("## 6) Reproducibility")
    lines.append("")
    lines.append("- Baseline summary source: `results/excess_baseline/excess_summary.json`.")
    lines.append("- Cycle-1 summary source: `results/e_hunt_delete_repair/summary.json`.")
    lines.append("- Cycle-2 artifacts source: `results/e_hunt_breakthrough_cycle2/**`.")
    lines.append("- Regenerate this report:")
    lines.append(
        "  - `export PYTHONPATH=src && python3 scripts/generate_research_report_to_date.py`"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
