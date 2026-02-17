from __future__ import annotations

import argparse
from pathlib import Path

from sparse_ruler.research_report_cycle2 import generate_research_report_cycle2


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cycle-2 research markdown report")
    parser.add_argument(
        "--cycle1-summary",
        type=Path,
        default=Path("results/e_hunt_delete_repair/summary.json"),
    )
    parser.add_argument(
        "--cycle2-summary",
        type=Path,
        default=Path("results/e_hunt_breakthrough_cycle2/summary.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/research_report_cycle2.md"),
    )
    args = parser.parse_args()

    out = generate_research_report_cycle2(
        cycle1_summary_path=args.cycle1_summary,
        cycle2_summary_path=args.cycle2_summary,
        out_path=args.out,
    )
    print(out)


if __name__ == "__main__":
    main()
