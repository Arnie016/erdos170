from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _scan_cycle2_root(cycle2_root: Path) -> Dict:
    summary_path = cycle2_root / "summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        best_rows = []
        for target in summary.get("targets", []):
            row = target.get("best_overall", {})
            best_rows.append(
                {
                    "N": target.get("N"),
                    "m_try": target.get("m_try"),
                    "stage_best_missing_count": row.get("stage_best_missing_count"),
                    "stage_best_missing_list": row.get("stage_best_missing_list", []),
                    "stage": row.get("stage"),
                    "candidate_id": row.get("candidate_id"),
                    "deleted_mark": row.get("deleted_mark"),
                }
            )
        return {
            "mode": "final",
            "summary_exists": True,
            "summary_path": str(summary_path),
            "best_rows": best_rows,
            "best_file_count": len(list(cycle2_root.rglob("best.json"))),
            "candidate_file_count": len(list(cycle2_root.rglob("candidate_summary.json"))),
        }

    candidate_files = list(cycle2_root.rglob("candidate_summary.json"))
    best_files = list(cycle2_root.rglob("best.json"))
    best_by_n: Dict[int, Dict] = {}
    for path in candidate_files:
        try:
            row = _load_json(path)
        except Exception:
            continue
        n = row.get("N")
        missing = row.get("stage_best_missing_count")
        if n is None or missing is None:
            continue
        n = int(n)
        missing = int(missing)
        current = best_by_n.get(n)
        if current is None or missing < current["stage_best_missing_count"]:
            best_by_n[n] = {
                "N": n,
                "m_try": row.get("m_try"),
                "stage_best_missing_count": missing,
                "stage_best_missing_list": row.get("stage_best_missing_list", []),
                "stage": row.get("stage"),
                "candidate_id": row.get("candidate_id"),
                "deleted_mark": row.get("deleted_mark"),
            }

    return {
        "mode": "partial",
        "summary_exists": False,
        "summary_path": str(summary_path),
        "best_rows": [best_by_n[n] for n in sorted(best_by_n)],
        "best_file_count": len(best_files),
        "candidate_file_count": len(candidate_files),
    }


def _table_row(values: List[str]) -> str:
    return " & ".join(values) + r" \\"


def _as_list_str(values: List[int]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


def _build_tables_tex(
    baseline: Optional[Dict],
    cycle1: Optional[Dict],
    cycle2: Dict,
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append(r"\section*{Computational Snapshot}")
    lines.append(r"\textbf{Generated (UTC):} " + _now_iso() + r"\\")
    lines.append("")

    if baseline:
        ratio = baseline.get("ratio", {})
        r = baseline.get("range", {})
        lines.append(r"\subsection*{Deterministic Baseline}")
        lines.append(
            r"Sweep: $N="
            + str(r.get("start"))
            + r"\ldots"
            + str(r.get("end"))
            + r"$ step "
            + str(r.get("step"))
            + r" ("
            + str(r.get("count"))
            + r" points)."
        )
        lines.append(
            r" Ratio range for $m/\sqrt{N}$: "
            + f"{ratio.get('min', 0.0):.6f}"
            + " to "
            + f"{ratio.get('max', 0.0):.6f}"
            + "."
        )
        lines.append("")
        lines.append(r"\begin{tabular}{rrrrr}")
        lines.append(r"\hline")
        lines.append(_table_row(["N", "m", "base", "E", "missing"]))
        lines.append(r"\hline")
        for row in baseline.get("fixed_points", []):
            lines.append(
                _table_row(
                    [
                        str(row.get("N")),
                        str(row.get("m")),
                        str(row.get("base_term")),
                        str(row.get("E")),
                        str(row.get("missing_count")),
                    ]
                )
            )
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append("")

    if cycle1:
        lines.append(r"\subsection*{Cycle-1 Best Results}")
        lines.append(r"\begin{tabular}{rrrrl}")
        lines.append(r"\hline")
        lines.append(_table_row(["N", "m\_try", "best\_missing", "deleted", "missing\_list"]))
        lines.append(r"\hline")
        for target in cycle1.get("targets", []):
            best = target.get("best_overall", {})
            lines.append(
                _table_row(
                    [
                        str(target.get("N")),
                        str(target.get("m_try")),
                        str(best.get("stage_best_missing_count")),
                        str(best.get("deleted_mark")),
                        r"\texttt{"
                        + _as_list_str(best.get("stage_best_missing_list", []))
                        + "}",
                    ]
                )
            )
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append("")

    lines.append(r"\subsection*{Cycle-2 Progress}")
    lines.append(
        r"Mode: "
        + cycle2.get("mode", "unknown")
        + r", best.json count="
        + str(cycle2.get("best_file_count", 0))
        + r", candidate\_summary count="
        + str(cycle2.get("candidate_file_count", 0))
        + "."
    )
    lines.append("")
    lines.append(r"\begin{tabular}{rrrrl}")
    lines.append(r"\hline")
    lines.append(_table_row(["N", "m\_try", "best\_missing", "deleted", "missing\_list"]))
    lines.append(r"\hline")
    for row in cycle2.get("best_rows", []):
        lines.append(
            _table_row(
                [
                    str(row.get("N")),
                    str(row.get("m_try")),
                    str(row.get("stage_best_missing_count")),
                    str(row.get("deleted_mark")),
                    r"\texttt{" + _as_list_str(row.get("stage_best_missing_list", [])) + "}",
                ]
            )
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_paper_artifacts(repo_root: Path) -> Tuple[Path, Dict]:
    results_dir = repo_root / "results"
    generated_dir = repo_root / "paper" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = results_dir / "excess_baseline" / "excess_summary.json"
    cycle1_path = results_dir / "e_hunt_delete_repair" / "summary.json"
    cycle2_root = results_dir / "e_hunt_breakthrough_cycle2"

    baseline = _load_json(baseline_path) if baseline_path.exists() else None
    cycle1 = _load_json(cycle1_path) if cycle1_path.exists() else None
    cycle2 = _scan_cycle2_root(cycle2_root) if cycle2_root.exists() else {
        "mode": "missing",
        "summary_exists": False,
        "summary_path": str(cycle2_root / "summary.json"),
        "best_rows": [],
        "best_file_count": 0,
        "candidate_file_count": 0,
    }

    copied = {
        "research_report_cycle1_md": _copy_if_exists(
            results_dir / "research_report_cycle1.md",
            generated_dir / "research_report_cycle1.md",
        ),
        "research_report_cycle2_md": _copy_if_exists(
            results_dir / "research_report_cycle2.md",
            generated_dir / "research_report_cycle2.md",
        ),
        "research_report_to_date_md": _copy_if_exists(
            results_dir / "research_report_to_date.md",
            generated_dir / "research_report_to_date.md",
        ),
        "baseline_summary_json": _copy_if_exists(
            baseline_path,
            generated_dir / "excess_summary.json",
        ),
        "cycle1_summary_json": _copy_if_exists(
            cycle1_path,
            generated_dir / "cycle1_summary.json",
        ),
    }

    _build_tables_tex(
        baseline=baseline,
        cycle1=cycle1,
        cycle2=cycle2,
        out_path=generated_dir / "tables.tex",
    )

    manifest = {
        "generated_at_utc": _now_iso(),
        "paths": {
            "baseline_summary": str(baseline_path),
            "cycle1_summary": str(cycle1_path),
            "cycle2_root": str(cycle2_root),
            "generated_dir": str(generated_dir),
        },
        "copied": copied,
        "cycle2_progress": cycle2,
    }
    manifest_path = generated_dir / "status_snapshot.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return generated_dir, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Overleaf-ready paper artifacts from repo data")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root containing results/, scripts/, and paper/",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    generated_dir, manifest = build_paper_artifacts(repo_root)
    print(json.dumps({"generated_dir": str(generated_dir), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
