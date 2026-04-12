from __future__ import annotations

import argparse
from pathlib import Path

from sparse_ruler.frontier_atlas import generate_frontier_atlas


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a structural frontier atlas")
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
    parser.add_argument("--out", type=Path, default=Path("results/frontier_atlas.md"))
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/frontier_atlas.json"),
    )
    args = parser.parse_args()

    out = generate_frontier_atlas(
        cycle1_summary_path=args.cycle1_summary,
        cycle2_summary_path=args.cycle2_summary,
        out_path=args.out,
        json_out_path=args.json_out,
    )
    print(out)


if __name__ == "__main__":
    main()
