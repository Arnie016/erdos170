from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import compute_W, format_csv_header, summarize_metrics
from .search import (
    default_large_scan,
    default_small_scan,
    run_experiments,
    write_best_solutions,
    write_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sparse ruler rigidity experiment suite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run annealing for a specific N,m")
    run_parser.add_argument("N", type=int)
    run_parser.add_argument("m", type=int)
    run_parser.add_argument("--runs", type=int, default=200)
    run_parser.add_argument("--steps", type=int, default=200_000)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--processes", type=int, default=1)
    run_parser.add_argument("--out-dir", type=Path, default=Path("outputs"))

    small_parser = subparsers.add_parser("small-scan", help="run default small scan")
    small_parser.add_argument("--runs", type=int, default=200)
    small_parser.add_argument("--steps", type=int, default=200_000)
    small_parser.add_argument("--seed", type=int, default=None)
    small_parser.add_argument("--processes", type=int, default=1)
    small_parser.add_argument("--out-dir", type=Path, default=Path("outputs"))

    large_parser = subparsers.add_parser("large-scan", help="run default large scan")
    large_parser.add_argument("--runs", type=int, default=200)
    large_parser.add_argument("--steps", type=int, default=200_000)
    large_parser.add_argument("--seed", type=int, default=None)
    large_parser.add_argument("--processes", type=int, default=1)
    large_parser.add_argument("--out-dir", type=Path, default=Path("outputs"))

    metrics_parser = subparsers.add_parser("metrics", help="compute metrics for a given ruler")
    metrics_parser.add_argument("N", type=int)
    metrics_parser.add_argument("A", type=str, help="comma-separated mark list")

    return parser


def handle_run(args: argparse.Namespace) -> int:
    results, impossible = run_experiments(
        N=args.N,
        m=args.m,
        runs=args.runs,
        steps=args.steps,
        seed=args.seed,
        processes=args.processes,
    )
    out_dir = args.out_dir
    if impossible:
        csv_path = out_dir / f"results_N{args.N}_m{args.m}.csv"
        write_csv([impossible.to_csv_row()], csv_path)
        return 0
    write_best_solutions(results, out_dir / "solutions", args.N, args.m)
    best = min(results, key=lambda r: r.H)
    csv_path = out_dir / f"results_N{args.N}_m{args.m}.csv"
    write_csv([best.metrics.to_csv_row()], csv_path)
    return 0


def handle_small_scan(args: argparse.Namespace) -> int:
    default_small_scan(args.out_dir, args.runs, args.steps, args.seed, args.processes)
    return 0


def handle_large_scan(args: argparse.Namespace) -> int:
    default_large_scan(args.out_dir, args.runs, args.steps, args.seed, args.processes)
    return 0


def handle_metrics(args: argparse.Namespace) -> int:
    A = [int(x.strip()) for x in args.A.split(",") if x.strip()]
    W = compute_W(A, args.N)
    metrics = summarize_metrics(args.N, A, W)
    payload = {
        "A": A,
        "metrics": metrics.__dict__,
        "csv_header": format_csv_header(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        return handle_run(args)
    if args.command == "small-scan":
        return handle_small_scan(args)
    if args.command == "large-scan":
        return handle_large_scan(args)
    if args.command == "metrics":
        return handle_metrics(args)
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
