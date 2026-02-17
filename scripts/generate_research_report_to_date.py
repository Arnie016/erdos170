from __future__ import annotations

import argparse
from pathlib import Path

from sparse_ruler.research_report_to_date import generate_research_report_to_date


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interim Erdos #170 report from completed data and live artifacts"
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
        "--out",
        type=Path,
        default=Path("results/research_report_to_date.md"),
    )
    args = parser.parse_args()

    out = generate_research_report_to_date(
        baseline_summary_path=args.baseline_summary,
        cycle1_summary_path=args.cycle1_summary,
        cycle2_root=args.cycle2_root,
        cycle2_config_path=args.cycle2_config,
        out_path=args.out,
    )
    print(out)


if __name__ == "__main__":
    main()
