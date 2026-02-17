from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_ruler.excess import parse_fixed_points, run_excess_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Wichmann excess baseline report")
    parser.add_argument("--start", type=int, default=500)
    parser.add_argument("--end", type=int, default=20000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument(
        "--fixed-points",
        nargs="+",
        default=["500,700,1000"],
        help="Comma-separated and/or space-separated list of fixed N points",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/excess_baseline"),
    )
    args = parser.parse_args()

    fixed = parse_fixed_points(args.fixed_points)
    summary = run_excess_baseline(
        start=args.start,
        end=args.end,
        step=args.step,
        fixed_points=fixed,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
