from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Sequence, Tuple

from sparse_ruler.metrics import compute_W, summarize_metrics
from sparse_ruler.search import refine_solution, run_experiments


def parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    return seeds


def summarize_best_incomplete(results):
    best = min(results, key=lambda r: r.metrics.missing_count)
    W = compute_W(best.A, best.metrics.N)
    missing = [d for d in range(1, best.metrics.N + 1) if W[d] == 0]
    return best, missing


def spike_mass_from_W(W: Sequence[int]) -> int:
    return sum(max(w - 2, 0) for w in W[1:])


def d3_size_from_W(W: Sequence[int]) -> int:
    return sum(1 for w in W[1:] if w >= 3)


def run_job_a(args: argparse.Namespace) -> None:
    seeds = parse_seeds(args.seeds)
    out_dir = args.out_dir / "jobA"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        "seed,best_M50,best_missing_count,complete_found,peak,gap,spike_mass,D3_size,ladder_proposals,ladder_accepts"
    ]
    best_overall_missing = None
    best_overall_missing_list: List[int] = []
    best_overall_result = None
    best_complete = None
    for seed in seeds:
        results, impossible = run_experiments(
            N=args.N,
            m=args.m,
            runs=args.runs,
            steps=args.steps,
            seed=seed,
            processes=args.processes,
        )
        if impossible:
            rows.append(f"{seed},impossible,0,,,,")
            continue
        best_missing_result, missing_list = summarize_best_incomplete(results)
        missing_count = best_missing_result.metrics.missing_count
        best_kernel = None
        best_kernel_result = None
        best_kernel_A = None
        for result in results:
            candidate_A = result.kernel_A if result.kernel_A is not None else result.A
            if result.kernel_M50 is not None:
                kernel_missing = result.kernel_M50
            else:
                W = compute_W(candidate_A, args.N)
                kernel_missing = sum(1 for d in range(1, min(50, args.N) + 1) if W[d] == 0)
            if best_kernel is None or kernel_missing < best_kernel:
                best_kernel = kernel_missing
                best_kernel_result = result
                best_kernel_A = candidate_A
        ladder_proposals = sum(r.ladder_proposals for r in results)
        ladder_accepts = sum(r.ladder_accepts for r in results)
        complete = [r for r in results if r.metrics.missing_count == 0]
        if complete:
            best_complete_seed = min(
                complete,
                key=lambda r: (r.metrics.E - r.metrics.Emin, r.metrics.peak, r.metrics.count_w3 + r.metrics.count_w4 + r.metrics.count_w5 + r.metrics.count_w6plus),
            )
            W_complete = compute_W(best_complete_seed.A, args.N)
            spike_mass = spike_mass_from_W(W_complete)
            d3_size = d3_size_from_W(W_complete)
            rows.append(
                f"{seed},{best_kernel},{missing_count},1,{best_complete_seed.metrics.peak},{best_complete_seed.metrics.E - best_complete_seed.metrics.Emin},"
                f"{spike_mass},{d3_size},{ladder_proposals},{ladder_accepts}"
            )
            if best_complete is None:
                best_complete = best_complete_seed
            else:
                best_complete = min(
                    [best_complete, best_complete_seed],
                    key=lambda r: (r.metrics.E - r.metrics.Emin, r.metrics.peak),
                )
        else:
            rows.append(f"{seed},{best_kernel},{missing_count},0,,,,,{ladder_proposals},{ladder_accepts}")
        if best_kernel_result is not None:
            kernel_A = best_kernel_A if best_kernel_A is not None else best_kernel_result.A
            W_kernel = compute_W(kernel_A, args.N)
            kernel_missing_count = sum(1 for d in range(1, args.N + 1) if W_kernel[d] == 0)
            payload = {
                "N": args.N,
                "m": args.m,
                "seed": seed,
                "A": kernel_A,
                "missing_count": kernel_missing_count,
                "M50": best_kernel,
            }
            (out_dir / f"best_seed_{seed}.json").write_text(json.dumps(payload, indent=2))
        if best_overall_missing is None or missing_count < best_overall_missing:
            best_overall_missing = missing_count
            best_overall_missing_list = missing_list
            best_overall_result = best_missing_result

    (out_dir / "summary.csv").write_text("\n".join(rows) + "\n")
    if best_complete:
        payload = {
            "N": args.N,
            "m": args.m,
            "A": best_complete.A,
            "metrics": best_complete.metrics.__dict__,
        }
        (out_dir / "best_complete.json").write_text(json.dumps(payload, indent=2))
    if best_overall_missing is not None:
        payload = {
            "best_missing_count": best_overall_missing,
            "missing_distances": best_overall_missing_list[:30],
            "A": best_overall_result.A if best_overall_result is not None else None,
        }
        (out_dir / "best_missing.json").write_text(json.dumps(payload, indent=2))


def run_job_b(args: argparse.Namespace) -> None:
    out_dir = args.out_dir / "jobB"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    data = json.loads(Path(args.warm_start).read_text())
    A = data["A"]

    best = None
    for seed in seeds:
        rng = random.Random(seed)
        result = refine_solution(
            N=args.N,
            A=A,
            steps=args.steps,
            rng=rng,
        )
        if best is None:
            best = result
        else:
            best = min(best, result, key=lambda r: (r.metrics.E - r.metrics.Emin, r.metrics.peak))

    if best:
        payload = {
            "N": args.N,
            "m": args.m,
            "A": best.A,
            "metrics": best.metrics.__dict__,
        }
        (out_dir / "best_complete.json").write_text(json.dumps(payload, indent=2))


def replace_worst_spike(
    N: int,
    A: Sequence[int],
    iterations: int,
    candidates: int,
    rng: random.Random,
) -> dict:
    current = list(sorted(A))

    def compute_W_local(A_list: Sequence[int]) -> Sequence[int]:
        return compute_W(A_list, N)

    def spike_distances(W: Sequence[int]) -> List[int]:
        return [d for d in range(1, N + 1) if W[d] >= 4]

    def spike_participation(W: Sequence[int], A_list: Sequence[int]) -> List[int]:
        spikes = set(spike_distances(W))
        scores = [0] * len(A_list)
        for i in range(len(A_list)):
            for j in range(i + 1, len(A_list)):
                d = abs(A_list[j] - A_list[i])
                if d in spikes:
                    scores[i] += 1
                    scores[j] += 1
        return scores

    def candidate_positions() -> List[int]:
        bias = list(range(1, 21)) + list(range(N - 20, N))
        mids = [rng.randint(1, N - 1) for _ in range(candidates)]
        pool = bias + mids
        rng.shuffle(pool)
        return pool[:candidates]

    best_payload = None
    for _ in range(iterations):
        W = compute_W_local(current)
        scores = spike_participation(W, current)
        worst_idx = max(range(1, len(current) - 1), key=lambda i: scores[i])
        base = current[:]
        base.pop(worst_idx)
        used = set(base)
        best_local = None
        for candidate in candidate_positions():
            if candidate in used or candidate <= 0 or candidate >= N:
                continue
            trial = sorted(base + [candidate])
            W_trial = compute_W_local(trial)
            metrics = summarize_metrics(N, trial, W_trial)
            if metrics.missing_count != 0:
                continue
            gap = metrics.E - metrics.Emin
            key = (gap, metrics.peak, sum(1 for w in W_trial[1:] if w >= 3))
            if best_local is None or key < best_local[0]:
                best_local = (key, trial, metrics)
        if best_local is not None:
            current = best_local[1]
            payload = {"A": current, "metrics": best_local[2].__dict__}
            best_payload = payload
    return best_payload or {"A": current, "metrics": summarize_metrics(N, current, compute_W_local(current)).__dict__}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    job_a = subparsers.add_parser("job-a")
    job_a.add_argument("--N", type=int, default=200)
    job_a.add_argument("--m", type=int, default=26)
    job_a.add_argument("--runs", type=int, default=200)
    job_a.add_argument("--steps", type=int, default=200000)
    job_a.add_argument("--seeds", type=str, default="1-20")
    job_a.add_argument("--processes", type=int, default=1)
    job_a.add_argument("--out-dir", type=Path, default=Path("outputs/fresh_runs"))

    job_b = subparsers.add_parser("job-b")
    job_b.add_argument("--N", type=int, default=100)
    job_b.add_argument("--m", type=int, default=18)
    job_b.add_argument("--steps", type=int, default=200000)
    job_b.add_argument("--seeds", type=str, default="1-20")
    job_b.add_argument("--warm-start", type=Path, required=True)
    job_b.add_argument("--out-dir", type=Path, default=Path("outputs/fresh_runs"))

    job_b_replace = subparsers.add_parser("job-b-replace")
    job_b_replace.add_argument("--N", type=int, default=100)
    job_b_replace.add_argument("--m", type=int, default=18)
    job_b_replace.add_argument("--iterations", type=int, default=100)
    job_b_replace.add_argument("--candidates", type=int, default=200)
    job_b_replace.add_argument("--seed", type=int, default=1)
    job_b_replace.add_argument("--warm-start", type=Path, required=True)
    job_b_replace.add_argument("--out-dir", type=Path, default=Path("outputs/fresh_runs_full"))

    args = parser.parse_args()
    if args.command == "job-a":
        run_job_a(args)
    elif args.command == "job-b":
        run_job_b(args)
    else:
        data = json.loads(Path(args.warm_start).read_text())
        payload = replace_worst_spike(
            N=args.N,
            A=data["A"],
            iterations=args.iterations,
            candidates=args.candidates,
            rng=random.Random(args.seed),
        )
        out_dir = args.out_dir / "jobB_replace"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "best_complete.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
