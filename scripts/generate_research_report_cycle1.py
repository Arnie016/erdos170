from __future__ import annotations

import argparse
from pathlib import Path

from sparse_ruler.research_report import generate_research_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cycle-1 research markdown report")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("results/excess_baseline/excess_summary.json"),
    )
    parser.add_argument(
        "--e-hunt-summary",
        type=Path,
        default=Path("results/e_hunt_delete_repair/summary.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/research_report_cycle1.md"),
    )
    args = parser.parse_args()

    out = generate_research_report(
        baseline_summary_path=args.baseline_summary,
        e_hunt_summary_path=args.e_hunt_summary,
        out_path=args.out,
    )
    print(out)


if __name__ == "__main__":
    main()
