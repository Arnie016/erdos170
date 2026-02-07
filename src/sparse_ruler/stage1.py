from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .metrics import compute_W
from .search import RulerState


@dataclass
class Stage1Diagnostics:
    accept_rate: float
    repair_accepts: int
    random_accepts: int
    coupled_accepts: int


@dataclass
class Stage1Best:
    marks: List[int]
    missing_count: int
    M25: int
    M50: int
    missing_distances: List[int]
    weighted_missing_sum: int
    diagnostics: Stage1Diagnostics
    hotspots_top20: List[Dict[str, int]]
    iterations_run: int


def run_stage1_scaffold(config: Dict) -> Dict[str, object]:
    base_problem = config["base_problem"]
    N = base_problem["N"]
    m = base_problem["m"]
    init_marks = list(config["init"]["initial_marks"])
    phase = config["phase_plan"][0]
    reporting = config["reporting"]

    seeds = phase["seeds"]
    iterations = phase["iterations"]
    freeze_policy = phase["freeze_policy"]
    move_policy = phase["move_policy"]
    acceptance = phase["acceptance"]
    escape_hatch = phase["escape_hatch"]
    outputs = phase["outputs"]

    output_root = Path(outputs["root_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    best_missing_overall = None
    best_seed = None
    best_results: List[Stage1Best] = []
    complete_hits = 0

    for seed in seeds:
        rng = random.Random(seed)
        result = run_stage1_seed(
            N=N,
            m=m,
            init_marks=init_marks,
            iterations=iterations,
            rng=rng,
            freeze_policy=freeze_policy,
            move_policy=move_policy,
            acceptance=acceptance,
            escape_hatch=escape_hatch,
            outputs=outputs,
            output_root=output_root,
            seed=seed,
        )
        best_results.append(result)
        if result.missing_count == 0:
            complete_hits += 1
        if best_missing_overall is None or result.missing_count < best_missing_overall:
            best_missing_overall = result.missing_count
            best_seed = seed

    median_best_missing = sorted(r.missing_count for r in best_results)[len(best_results) // 2]
    summary_payload = {
        "best_missing_overall": best_missing_overall,
        "median_best_missing": median_best_missing,
        "complete_hits": complete_hits,
        "best_seed": best_seed,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    return {"summary_path": summary_path, "per_seed_paths": seeds, "reporting_fields": reporting}


def run_stage1_seed(
    *,
    N: int,
    m: int,
    init_marks: Sequence[int],
    iterations: int,
    rng: random.Random,
    freeze_policy: Dict,
    move_policy: Dict,
    acceptance: Dict,
    escape_hatch: Dict,
    outputs: Dict,
    output_root: Path,
    seed: int,
) -> Stage1Best:
    state = RulerState(N, init_marks)
    if len(state.A) != m:
        raise ValueError("initial marks length does not match m")

    frozen = build_frozen_set(state.A, freeze_policy, move_policy, state.W)
    temperature = acceptance["metropolis"]["T0"]
    cooling = acceptance["metropolis"]["alpha"]
    accept_equal_prob = acceptance["metropolis"]["accept_equal_prob"]
    kernel_guard = acceptance["kernel_guard"]

    best_state = RulerState(N, state.A)
    min_d = move_policy["repair_mode"].get("target_missing_min_d", 1)
    best_weighted = weighted_missing_sum(state.W, N, min_d=min_d)
    best_missing = state.missing_count
    best_step = 0

    accept_total = 0
    repair_accepts = 0
    random_accepts = 0
    coupled_accepts = 0

    trace: List[Dict[str, int]] = []

    steps_run = 0
    for step in range(1, iterations + 1):
        steps_run = step
        use_coupled = should_use_coupled_move(
            escape_hatch,
            step,
            best_step,
            best_missing,
        )
        mode = choose_mode(rng, move_policy, use_coupled, escape_hatch)
        if mode == "coupled":
            moved, old_state = propose_coupled_move(state, rng, move_policy, frozen)
        elif mode == "repair":
            moved, old_state = propose_repair_move(state, rng, move_policy, frozen)
        else:
            moved, old_state = propose_random_move(state, rng, move_policy, frozen)
        if not moved:
            temperature *= cooling
            continue

        new_weighted = weighted_missing_sum(state.W, N, min_d=min_d)
        new_M25 = missing_count_range(state.W, N, 25)
        new_M50 = missing_count_range(state.W, N, 50)
        old_weighted = weighted_missing_sum(old_state.W, N, min_d=min_d)
        old_M25 = missing_count_range(old_state.W, N, 25)
        old_M50 = missing_count_range(old_state.W, N, 50)

        accept = evaluate_acceptance(
            new_weighted,
            old_weighted,
            new_M25,
            old_M25,
            new_M50,
            old_M50,
            temperature,
            rng,
            kernel_guard,
            accept_equal_prob,
        )
        if accept:
            accept_total += 1
            if mode == "repair":
                repair_accepts += 1
            elif mode == "random":
                random_accepts += 1
            else:
                coupled_accepts += 1
            if new_weighted < best_weighted:
                best_weighted = new_weighted
                best_missing = state.missing_count
                best_state = RulerState(N, state.A)
                best_step = step
        else:
            state = old_state

        temperature *= cooling

        if outputs.get("save_per_seed_trace"):
            current_M25 = missing_count_range(state.W, N, 25)
            current_M50 = missing_count_range(state.W, N, 50)
            trace.append(
                {
                    "step": step,
                    "missing_count": state.missing_count,
                    "M25": current_M25,
                    "M50": current_M50,
                    "weighted_missing_sum": weighted_missing_sum(state.W, N, min_d=min_d),
                }
            )
        if outputs.get("stop_early", {}).get("enabled") and state.missing_count == 0:
            break

    diagnostics = Stage1Diagnostics(
        accept_rate=accept_total / max(steps_run, 1),
        repair_accepts=repair_accepts,
        random_accepts=random_accepts,
        coupled_accepts=coupled_accepts,
    )
    missing = missing_list(best_state.W, N)
    hotspots = build_hotspots(missing, top_k=20)
    best_payload = Stage1Best(
        marks=best_state.A,
        missing_count=best_state.missing_count,
        M25=missing_count_range(best_state.W, N, 25),
        M50=missing_count_range(best_state.W, N, 50),
        missing_distances=missing_list(best_state.W, N),
        weighted_missing_sum=best_weighted,
        diagnostics=diagnostics,
        hotspots_top20=hotspots,
        iterations_run=steps_run,
    )
    write_seed_outputs(output_root, seed, best_payload, trace, outputs)
    return best_payload


def build_frozen_set(
    marks: Sequence[int],
    freeze_policy: Dict,
    move_policy: Dict,
    W: Sequence[int],
) -> set[int]:
    frozen = set(freeze_policy.get("always_freeze", []))
    freeze_low_le = freeze_policy.get("freeze_low_le")
    if freeze_low_le is not None:
        frozen.update(mark for mark in marks if mark <= freeze_low_le)
    top_support = freeze_policy.get("also_freeze_top_scaffold_by_support", 0)
    if top_support:
        scaffold_threshold = move_policy.get("scaffold_threshold", 0)
        support_max_d = freeze_policy.get("support_range_max_d", 50)
        scores = unique_support_scores(marks, W, support_max_d)
        candidates = [mark for mark in marks if mark <= scaffold_threshold]
        candidates.sort(key=lambda mark: scores.get(mark, 0), reverse=True)
        frozen.update(candidates[:top_support])
    return frozen


def unique_support_scores(marks: Sequence[int], W: Sequence[int], max_d: int) -> Dict[int, int]:
    scores = {mark: 0 for mark in marks}
    for i, ai in enumerate(marks):
        for j in range(i + 1, len(marks)):
            aj = marks[j]
            d = abs(aj - ai)
            if d <= max_d and W[d] == 1:
                scores[ai] += 1
                scores[aj] += 1
    return scores


def choose_mode(rng: random.Random, move_policy: Dict, use_coupled: bool, escape_hatch: Dict) -> str:
    if use_coupled and rng.random() < escape_hatch["proposal_prob_when_enabled"]:
        return "coupled"
    mix = move_policy["proposal_mix"]
    if rng.random() < mix["repair_mode_prob"]:
        return "repair"
    return "random"


def propose_repair_move(
    state: RulerState,
    rng: random.Random,
    move_policy: Dict,
    frozen: set[int],
) -> Tuple[bool, RulerState]:
    repair_mode = move_policy["repair_mode"]
    min_d = repair_mode.get("target_missing_min_d", 1)
    missing = [d for d in range(min_d, state.N + 1) if state.W[d] == 0]
    if repair_mode.get("target_missing_only") and not missing:
        return False, state
    if missing:
        weights = [d * d for d in missing]
        target_d = weighted_choice(rng, missing, weights)
    else:
        target_d = rng.randint(min_d, state.N)
    anchors = available_anchors(state, move_policy, frozen)
    candidates = build_candidates(state, anchors, target_d)
    if not candidates:
        return False, state
    idx = select_movable_mark(state, move_policy, frozen, rng)
    if idx is None:
        return False, state
    old_state = RulerState(state.N, state.A)
    candidate = rng.choice(candidates)
    state.move_mark(idx, candidate)
    return True, old_state


def propose_random_move(
    state: RulerState,
    rng: random.Random,
    move_policy: Dict,
    frozen: set[int],
) -> Tuple[bool, RulerState]:
    sample_min, sample_max = move_policy["random_mode"]["sample_range"]
    idx = select_movable_mark(state, move_policy, frozen, rng)
    if idx is None:
        return False, state
    occupied = set(state.A)
    for _ in range(20):
        candidate = rng.randint(sample_min, sample_max)
        if candidate in occupied:
            continue
        old_state = RulerState(state.N, state.A)
        state.move_mark(idx, candidate)
        return True, old_state
    return False, state


def propose_coupled_move(
    state: RulerState,
    rng: random.Random,
    move_policy: Dict,
    frozen: set[int],
) -> Tuple[bool, RulerState]:
    repair_mode = move_policy["repair_mode"]
    min_d = repair_mode.get("target_missing_min_d", 1)
    missing = [d for d in range(min_d, state.N + 1) if state.W[d] == 0]
    if not missing:
        return False, state
    weights = [d * d for d in missing]
    target_d = weighted_choice(rng, missing, weights)
    anchors = available_anchors(state, move_policy, frozen)
    candidates = build_candidates(state, anchors, target_d)
    if not candidates:
        return False, state
    idx1 = select_movable_mark(state, move_policy, frozen, rng)
    if idx1 is None:
        return False, state
    old_state = RulerState(state.N, state.A)
    candidate1 = rng.choice(candidates)
    state.move_mark(idx1, candidate1)

    if missing_count_range(state.W, state.N, 25) > 0 or missing_count_range(state.W, state.N, 50) > 2:
        secondary_missing = [d for d in range(1, 51) if state.W[d] == 0]
        if not secondary_missing:
            return False, old_state
        target2 = max(secondary_missing)
        anchors2 = available_anchors(state, move_policy, frozen)
        candidates2 = build_candidates(state, anchors2, target2)
        if not candidates2:
            return False, old_state
        idx2 = select_movable_mark(state, move_policy, frozen, rng, exclude={idx1})
        if idx2 is None:
            return False, old_state
        candidate2 = rng.choice(candidates2)
        state.move_mark(idx2, candidate2)
    return True, old_state


def available_anchors(state: RulerState, move_policy: Dict, frozen: set[int]) -> List[int]:
    anchors = move_policy["repair_mode"]["anchors"]
    le_value = anchors["le_value"]
    pool = [mark for mark in state.A if mark <= le_value]
    pool.extend(mark for mark in frozen if mark not in pool)
    for mark in state.A:
        if mark not in pool and mark <= move_policy.get("scaffold_threshold", le_value):
            pool.append(mark)
    return list(dict.fromkeys(pool))


def build_candidates(state: RulerState, anchors: Iterable[int], target_d: int) -> List[int]:
    occupied = set(state.A)
    candidates = set()
    for anchor in anchors:
        for candidate in (anchor + target_d, anchor - target_d):
            if 1 <= candidate <= state.N - 1 and candidate not in occupied:
                candidates.add(candidate)
    return list(candidates)


def select_movable_mark(
    state: RulerState,
    move_policy: Dict,
    frozen: set[int],
    rng: random.Random,
    exclude: Optional[set[int]] = None,
) -> Optional[int]:
    scaffold_threshold = move_policy["scaffold_threshold"]
    candidate_indices = [
        idx
        for idx in range(1, len(state.A) - 1)
        if state.A[idx] > scaffold_threshold and state.A[idx] not in frozen
    ]
    if exclude:
        candidate_indices = [idx for idx in candidate_indices if idx not in exclude]
    if not candidate_indices:
        return None
    scores = responsibility_scores(state.A, state.W, move_policy["repair_mode"]["anchors"]["le_value"])
    ranked = sorted(candidate_indices, key=lambda idx: scores.get(state.A[idx], 0))
    top_k = move_policy["repair_mode"]["mark_selection"]["top_k_bias"]
    shortlist = ranked[: max(1, min(top_k, len(ranked)))]
    return rng.choice(shortlist)


def responsibility_scores(marks: Sequence[int], W: Sequence[int], max_d: int) -> Dict[int, int]:
    scores = {mark: 0 for mark in marks}
    for i, ai in enumerate(marks):
        for j in range(i + 1, len(marks)):
            aj = marks[j]
            d = abs(aj - ai)
            if d <= max_d and W[d] == 1:
                scores[ai] += 1
                scores[aj] += 1
    return scores


def weighted_missing_sum(W: Sequence[int], N: int, min_d: int) -> int:
    return sum(d * d for d in range(min_d, N + 1) if W[d] == 0)


def missing_list(W: Sequence[int], N: int) -> List[int]:
    return [d for d in range(1, N + 1) if W[d] == 0]


def missing_count_range(W: Sequence[int], N: int, max_d: int) -> int:
    return sum(1 for d in range(1, min(N, max_d) + 1) if W[d] == 0)


def evaluate_acceptance(
    new_weighted: int,
    old_weighted: int,
    new_M25: int,
    old_M25: int,
    new_M50: int,
    old_M50: int,
    temperature: float,
    rng: random.Random,
    kernel_guard: Dict,
    accept_equal_prob: float,
) -> bool:
    if kernel_guard.get("require_M25", 0) == 0 and new_M25 > 0:
        return False
    allow_M50 = kernel_guard.get("allow_M50_max", 0)
    penalty_factor = kernel_guard.get("M26_to_M50_penalty", 0)
    new_penalty = penalty_factor * max(0, new_M50 - allow_M50)
    old_penalty = penalty_factor * max(0, old_M50 - allow_M50)
    new_score = new_weighted + new_penalty
    old_score = old_weighted + old_penalty
    if new_score < old_score:
        return True
    if new_score == old_score:
        return rng.random() < accept_equal_prob
    delta = new_score - old_score
    if temperature <= 0:
        return False
    prob = math.exp(-delta / temperature)
    return rng.random() < prob


def should_use_coupled_move(
    escape_hatch: Dict,
    step: int,
    best_step: int,
    best_missing: int,
) -> bool:
    if not escape_hatch.get("enable_two_mark_moves"):
        return False
    trigger = escape_hatch["trigger"]
    if best_missing > trigger["missing_count_at_most"]:
        return False
    stall_steps = trigger["stall_steps"]
    if step - best_step < stall_steps:
        return False
    return True


def weighted_choice(rng: random.Random, values: Sequence[int], weights: Sequence[int]) -> int:
    total = sum(weights)
    r = rng.random() * total
    for value, weight in zip(values, weights):
        r -= weight
        if r <= 0:
            return value
    return values[-1]


def build_hotspots(missing: Sequence[int], top_k: int) -> List[Dict[str, int]]:
    ranked = sorted(missing, reverse=True)[:top_k]
    return [{"d": d, "weight": d * d} for d in ranked]


def write_seed_outputs(
    output_root: Path,
    seed: int,
    best: Stage1Best,
    trace: Sequence[Dict[str, int]],
    outputs: Dict,
) -> None:
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best_payload = {
        "seed": seed,
        "iterations_run": best.iterations_run,
        "best": {
            "missing_count": best.missing_count,
            "M25": best.M25,
            "M50": best.M50,
            "marks": best.marks,
            "missing_distances": best.missing_distances,
        },
        "diagnostics": {
            "accept_rate": best.diagnostics.accept_rate,
            "repair_accepts": best.diagnostics.repair_accepts,
            "random_accepts": best.diagnostics.random_accepts,
            "coupled_accepts": best.diagnostics.coupled_accepts,
        },
        "hotspots_top20": best.hotspots_top20,
    }
    if outputs.get("save_per_seed_best", True):
        (seed_dir / "best.json").write_text(json.dumps(best_payload, indent=2))
    if outputs.get("save_per_seed_trace"):
        trace_path = seed_dir / "trace.jsonl"
        with trace_path.open("w") as handle:
            for entry in trace:
                handle.write(json.dumps(entry))
                handle.write("\n")
    if outputs.get("save_missing_list"):
        (seed_dir / "missing_best.txt").write_text(json.dumps(best.missing_distances))
    if outputs.get("save_marks_best"):
        (seed_dir / "marks_best.txt").write_text(json.dumps(best.marks))
    if outputs.get("save_hotspot_table"):
        (seed_dir / "hotspots.json").write_text(json.dumps(best.hotspots_top20, indent=2))
