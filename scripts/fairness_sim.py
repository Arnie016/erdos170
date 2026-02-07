from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class PolicyResult:
    name: str
    d_max: int
    thrash_count: int
    e_total: int
    max_pair_peak: int
    defect_count: int


def anytime_maximin_policy(counts: List[int], k: int, rng: random.Random) -> List[int]:
    min_count = min(counts)
    candidates = [i for i, c in enumerate(counts) if c == min_count]
    rng.shuffle(candidates)
    return candidates[:k]


def planned_quota_policy(remaining: List[int], k: int, rng: random.Random) -> List[int]:
    max_remaining = max(remaining)
    candidates = [i for i, r in enumerate(remaining) if r == max_remaining]
    rng.shuffle(candidates)
    chosen = candidates[:]
    if len(chosen) < k:
        rest = [i for i, r in enumerate(remaining) if r != max_remaining]
        rest.sort(key=lambda i: (-remaining[i], i))
        chosen.extend(rest[: k - len(chosen)])
    return chosen[:k]


def round_robin_policy(pointer: int, m: int, k: int) -> List[int]:
    return [(pointer + offset) % m for offset in range(k)]


def build_cyclic_block(m: int, k: int) -> List[int]:
    block = [0]
    diff_counts = [0] * m
    while len(block) < k:
        best = None
        best_key = None
        for candidate in range(1, m):
            if candidate in block:
                continue
            updated = diff_counts[:]
            for existing in block:
                diff = (candidate - existing) % m
                if diff != 0:
                    updated[diff] += 1
            max_count = max(updated)
            sum_sq = sum(c * c for c in updated)
            key = (max_count, sum_sq, candidate)
            if best_key is None or key < best_key:
                best_key = key
                best = candidate
        if best is None:
            break
        for existing in block:
            diff = (best - existing) % m
            if diff != 0:
                diff_counts[diff] += 1
        block.append(best)
    return block


def simulate(
    m: int,
    k: int,
    t_rounds: int,
    policy: str,
    design_block: Sequence[int],
    rng: random.Random,
) -> Tuple[List[List[int]], int, int]:
    counts = [0] * m
    boost_times = [[] for _ in range(m)]
    thrash_count = 0
    last_min_set: Iterable[int] | None = None
    pointer = 0

    total_boosts = k * t_rounds
    base = total_boosts // m
    remainder = total_boosts % m
    target = [base + (1 if i < remainder else 0) for i in range(m)]
    remaining = target[:]

    d_max = 0
    for t in range(1, t_rounds + 1):
        if policy == "anytime_maximin":
            chosen = anytime_maximin_policy(counts, k, rng)
        elif policy == "end_of_horizon":
            chosen = planned_quota_policy(remaining, k, rng)
        elif policy == "round_robin":
            chosen = round_robin_policy(pointer, m, k)
            pointer = (pointer + k) % m
        elif policy == "design_based":
            chosen = [(t - 1 + offset) % m for offset in design_block]
        else:
            raise ValueError(f"Unknown policy: {policy}")

        for idx in chosen:
            counts[idx] += 1
            boost_times[idx].append(t)
            if policy == "end_of_horizon":
                remaining[idx] -= 1

        min_count = min(counts)
        max_count = max(counts)
        d_max = max(d_max, max_count - min_count)
        min_set = {i for i, c in enumerate(counts) if c == min_count}
        if last_min_set is not None and min_set != set(last_min_set):
            thrash_count += 1
        last_min_set = min_set

    return boost_times, d_max, thrash_count


def build_bitsets(boost_times: Sequence[Sequence[int]], t_rounds: int) -> List[int]:
    bitsets = []
    for times in boost_times:
        bits = 0
        for t in times:
            bits |= 1 << (t - 1)
        bitsets.append(bits)
    return bitsets


def pairwise_metrics(bitsets: Sequence[int], t_rounds: int) -> Tuple[int, int, int]:
    e_total = 0
    max_peak = 0
    defect_count = 0
    m = len(bitsets)
    for i in range(m - 1):
        a = bitsets[i]
        for j in range(i + 1, m):
            b = bitsets[j]
            pair_peak = 0
            pair_energy = 0
            pair_defects = 0
            for d in range(t_rounds):
                count = (a & (b << d)).bit_count()
                if count:
                    pair_energy += count * count
                    if count > pair_peak:
                        pair_peak = count
                    if count >= 3:
                        pair_defects += 1
            e_total += pair_energy
            max_peak = max(max_peak, pair_peak)
            defect_count += pair_defects
    return e_total, max_peak, defect_count


def histogram_for_pair(
    bitsets: Sequence[int],
    t_rounds: int,
    pair: Tuple[int, int] = (0, 1),
) -> Dict[int, int]:
    i, j = pair
    a = bitsets[i]
    b = bitsets[j]
    histogram: Dict[int, int] = {}
    for d in range(t_rounds):
        count = (a & (b << d)).bit_count()
        if count:
            histogram[d] = count
    return histogram


def run_simulation(
    m: int,
    k: int,
    t_rounds: int,
    seed: int,
) -> Tuple[List[PolicyResult], Dict[str, Dict[int, int]]]:
    policies = ["anytime_maximin", "end_of_horizon", "round_robin", "design_based"]
    design_block = build_cyclic_block(m, k)
    rng = random.Random(seed)

    results = []
    histograms = {}
    for policy in policies:
        boost_times, d_max, thrash_count = simulate(m, k, t_rounds, policy, design_block, rng)
        bitsets = build_bitsets(boost_times, t_rounds)
        e_total, max_pair_peak, defect_count = pairwise_metrics(bitsets, t_rounds)
        results.append(
            PolicyResult(
                name=policy,
                d_max=d_max,
                thrash_count=thrash_count,
                e_total=e_total,
                max_pair_peak=max_pair_peak,
                defect_count=defect_count,
            )
        )
        histograms[policy] = histogram_for_pair(bitsets, t_rounds)

    return results, histograms


def write_summary(results: Sequence[PolicyResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "D_max", "thrash_count", "E_total", "max_pair_peak", "defect_count"])
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.d_max,
                    result.thrash_count,
                    result.e_total,
                    result.max_pair_peak,
                    result.defect_count,
                ]
            )


def write_histograms(histograms: Dict[str, Dict[int, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["policy,diff,count"]
    for policy, data in histograms.items():
        for diff, count in sorted(data.items()):
            lines.append(f"{policy},{diff},{count}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=60)
    parser.add_argument("--boosts", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/fairness_sim"))
    args = parser.parse_args()

    results, histograms = run_simulation(args.agents, args.boosts, args.rounds, args.seed)
    write_summary(results, args.out_dir / "summary.csv")
    write_histograms(histograms, args.out_dir / "pair_histogram.csv")


if __name__ == "__main__":
    main()
