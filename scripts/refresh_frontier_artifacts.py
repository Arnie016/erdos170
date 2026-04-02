from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_ruler.frontier_artifacts import (
    render_exact_frontier_summary,
    render_rigidity_memo,
    scan_frontier_root,
    write_text_if_changed,
)
from sparse_ruler.research_report_to_date import generate_research_report_to_date


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh frontier evidence artifacts")
    parser.add_argument(
        "--frontier-root",
        type=Path,
        default=Path("results/e_hunt_breakthrough_cycle3_tailfocus"),
        help="Frontier run root to scan for live evidence",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("results/excess_baseline/excess_summary.json"),
    )
    parser.add_argument(
        "--cycle1-summary",
        type=Path,
        default=Path("results/e_hunt_delete_repair/summary.json"),
    )
    parser.add_argument(
        "--cycle2-root",
        type=Path,
        default=Path("results/e_hunt_breakthrough_cycle2"),
    )
    parser.add_argument(
        "--cycle2-config",
        type=Path,
        default=Path("configs/e_hunt_breakthrough_cycle2.json"),
    )
    parser.add_argument(
        "--research-report-out",
        type=Path,
        default=Path("results/research_report_to_date.md"),
    )
    parser.add_argument(
        "--exact-summary-out",
        type=Path,
        default=Path("results/exact_frontier_summary.md"),
    )
    parser.add_argument(
        "--rigidity-memo-out",
        type=Path,
        default=Path("results/rigidity_memo.md"),
    )
    parser.add_argument(
        "--snapshot-json-out",
        type=Path,
        default=Path("results/exact_frontier_summary.json"),
    )
    args = parser.parse_args()

    snapshot = scan_frontier_root(args.frontier_root)
    exact_summary = render_exact_frontier_summary(snapshot)
    rigidity_memo = render_rigidity_memo(snapshot)

    exact_written = write_text_if_changed(args.exact_summary_out, exact_summary)
    memo_written = write_text_if_changed(args.rigidity_memo_out, rigidity_memo)
    args.snapshot_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot_json_out.write_text(json.dumps(snapshot, indent=2) + "\n")

    report_out = generate_research_report_to_date(
        baseline_summary_path=args.baseline_summary,
        cycle1_summary_path=args.cycle1_summary,
        cycle2_root=args.cycle2_root,
        cycle2_config_path=args.cycle2_config,
        out_path=args.research_report_out,
    )

    print(
        json.dumps(
            {
                "frontier_root": str(args.frontier_root),
                "exact_summary_out": str(args.exact_summary_out),
                "rigidity_memo_out": str(args.rigidity_memo_out),
                "snapshot_json_out": str(args.snapshot_json_out),
                "research_report_out": str(report_out),
                "exact_summary_written": exact_written,
                "rigidity_memo_written": memo_written,
                "summary_exists": snapshot["summary_exists"],
                "best_file_count": snapshot["best_file_count"],
                "candidate_file_count": snapshot["candidate_file_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

