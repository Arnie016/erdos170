from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .metrics import compute_W
from .search import RulerState


@dataclass(frozen=True)
class ObjectiveScore:
    kernel_violation: int
    over_m25: int
    over_m50: int
    missing_count: int
    secondary: int
    M25: int
    M50: int

    def key(self) -> Tuple[int, int, int, int, int]:
        return (
            self.kernel_violation,
            self.over_m25,
            self.over_m50,
            self.missing_count,
            self.secondary,
        )


@dataclass(frozen=True)
class SeedOutcome:
    seed: int
    iterations_run: int
    objective_key: Tuple[int, int, int, int, int]
    best_missing_count: int
    best_secondary: int
    M25: int
    M50: int
    best_marks: List[int]
    best_missing_list: List[int]
    coupled_attempts: int
    coupled_accepts: int
    endpoint_window_attempts: int
    endpoint_window_accepts: int
    tail_trigger_count: int
    endpoint_hole_mode_fired: bool
    complete_independent: bool
    integrity_checks: Dict[str, bool]


STAGE_ORDER = ["screen", "deep", "championship"]


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _merge_dict(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = _json_clone(defaults)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_token(value: Any, N: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        token = value.strip().upper()
        if token == "N":
            return N
        if token.lstrip("-").isdigit():
            return int(token)

        # Accept symbolic offsets such as N-1, N-4, N+2.
        match = re.fullmatch(r"N([+-])(\d+)", token)
        if match:
            sign, offset_text = match.groups()
            offset = int(offset_text)
            return N + offset if sign == "+" else N - offset
    raise ValueError(f"unsupported token {value!r}")


def _resolve_int_list(values: Iterable[Any], N: int) -> List[int]:
    return [_resolve_token(value, N) for value in values]


def missing_distances_from_W(W: Sequence[int], N: int) -> List[int]:
    return [d for d in range(1, N + 1) if W[d] == 0]


def missing_count_upto(W: Sequence[int], upto: int) -> int:
    return sum(1 for d in range(1, min(len(W) - 1, upto) + 1) if W[d] == 0)


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


def weighted_choice(rng: random.Random, values: Sequence[int], weights: Sequence[int]) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.choice(list(values))
    r = rng.random() * total
    for value, weight in zip(values, weights):
        r -= weight
        if r <= 0:
            return value
    return values[-1]


def build_frozen_set(state: RulerState, freeze_policy: Dict[str, Any]) -> Set[int]:
    frozen: Set[int] = set(_resolve_int_list(freeze_policy.get("always_freeze", [0, "N"]), state.N))

    freeze_prefix_le = freeze_policy.get("freeze_prefix_le")
    if freeze_prefix_le is not None:
        frozen.update(mark for mark in state.A if mark <= int(freeze_prefix_le))

    support_cfg = freeze_policy.get("also_freeze_top_by_support", {})
    if support_cfg.get("enabled"):
        count = int(support_cfg.get("count", 0))
        max_d = int(support_cfg.get("range_max_d", 50))
        scores = unique_support_scores(state.A, state.W, max_d)
        candidates = [
            mark
            for mark in state.A
            if mark not in frozen and mark not in (0, state.N)
        ]
        candidates.sort(key=lambda mark: (scores.get(mark, 0), -mark), reverse=True)
        frozen.update(candidates[:count])

    frozen.add(0)
    frozen.add(state.N)
    return frozen


def compute_objective_score(state: RulerState, objective_cfg: Dict[str, Any]) -> ObjectiveScore:
    missing_power_p = int(objective_cfg.get("missing_power_p", 2))
    kernel_penalty_lambda = int(objective_cfg.get("kernel_penalty_lambda", 2000))
    kernel_guard = objective_cfg.get("kernel_guard", {})

    missing = missing_distances_from_W(state.W, state.N)
    weighted_missing_sum = sum(d**missing_power_p for d in missing)

    M25 = missing_count_upto(state.W, 25)
    M50 = missing_count_upto(state.W, 50)

    require_m25 = kernel_guard.get("require_M25")
    allow_m50_max = kernel_guard.get("allow_M50_max")

    over_m25 = 0
    if require_m25 is not None:
        over_m25 = max(0, M25 - int(require_m25))

    over_m50 = 0
    if allow_m50_max is not None:
        over_m50 = max(0, M50 - int(allow_m50_max))

    kernel_violation = 1 if (over_m25 > 0 or over_m50 > 0) else 0
    secondary = weighted_missing_sum + kernel_penalty_lambda * M50

    return ObjectiveScore(
        kernel_violation=kernel_violation,
        over_m25=over_m25,
        over_m50=over_m50,
        missing_count=state.missing_count,
        secondary=secondary,
        M25=M25,
        M50=M50,
    )


def build_anchors(state: RulerState, moves_cfg: Dict[str, Any]) -> List[int]:
    anchors_cfg = moves_cfg.get("anchors", {})
    le_value = int(anchors_cfg.get("le_value", state.N))
    top_high = int(anchors_cfg.get("top_high", 0))

    anchors = [mark for mark in state.A if mark <= le_value]
    high_marks = [mark for mark in sorted(state.A, reverse=True) if mark > le_value]
    for mark in high_marks[:top_high]:
        if mark not in anchors:
            anchors.append(mark)
    return anchors


def select_movable_index(
    state: RulerState,
    *,
    frozen: Set[int],
    moves_cfg: Dict[str, Any],
    rng: random.Random,
    exclude_marks: Optional[Set[int]] = None,
) -> Optional[int]:
    exclude_marks = exclude_marks or set()

    movable = [
        idx
        for idx in range(1, len(state.A) - 1)
        if state.A[idx] not in frozen and state.A[idx] not in exclude_marks
    ]
    if not movable:
        return None

    mark_cfg = moves_cfg.get("mark_selection", {})
    top_k_bias = int(mark_cfg.get("top_k_bias", 8))
    responsibility_max_d = int(mark_cfg.get("responsibility_max_d", 50))

    scores = unique_support_scores(state.A, state.W, responsibility_max_d)
    ranked = sorted(movable, key=lambda idx: (scores.get(state.A[idx], 0), state.A[idx]))

    k = max(1, min(top_k_bias, len(ranked)))
    return rng.choice(ranked[:k])


def build_candidate_positions(
    state: RulerState,
    anchors: Sequence[int],
    target_d: int,
    candidate_cfg: Dict[str, Any],
    rng: random.Random,
) -> List[int]:
    occupied = set(state.A)
    dedupe = bool(candidate_cfg.get("dedupe", True))
    seen: Set[int] = set()
    candidates: List[int] = []

    for anchor in anchors:
        for candidate in (anchor + target_d, anchor - target_d):
            if candidate <= 0 or candidate >= state.N:
                continue
            if candidate in occupied:
                continue
            if dedupe and candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

    pool_size = int(candidate_cfg.get("pool_size", 0))
    if pool_size > 0 and len(candidates) > pool_size:
        candidates = rng.sample(candidates, pool_size)

    return candidates


def choose_candidate_with_bias(
    candidates: Sequence[int],
    candidate_cfg: Dict[str, Any],
    N: int,
    rng: random.Random,
) -> Optional[int]:
    if not candidates:
        return None

    if candidate_cfg.get("bias_to_upper_half"):
        upper_half_prob = float(candidate_cfg.get("upper_half_prob", 0.75))
        if rng.random() < upper_half_prob:
            preferred = [c for c in candidates if c > N // 2]
            if preferred:
                return rng.choice(preferred)

    bias_cfg = candidate_cfg.get("bias_to_scaffold", {})
    if bias_cfg.get("enabled") and rng.random() < float(bias_cfg.get("prob", 0.0)):
        lo, hi = bias_cfg.get("prefer_range", [1, N - 1])
        preferred = [c for c in candidates if int(lo) <= c <= int(hi)]
        if preferred:
            return rng.choice(preferred)

    return rng.choice(list(candidates))


def apply_repair_move(
    state: RulerState,
    *,
    frozen: Set[int],
    moves_cfg: Dict[str, Any],
    objective_cfg: Dict[str, Any],
    rng: random.Random,
    exclude_marks: Optional[Set[int]] = None,
) -> Optional[Tuple[int, int]]:
    missing = missing_distances_from_W(state.W, state.N)
    if not missing:
        return None

    missing_power_p = int(objective_cfg.get("missing_power_p", 2))
    weights = [d**missing_power_p for d in missing]
    target_d = weighted_choice(rng, missing, weights)

    anchors = build_anchors(state, moves_cfg)
    candidate_cfg = moves_cfg.get("candidate_positions", {})
    candidates = build_candidate_positions(state, anchors, target_d, candidate_cfg, rng)
    if not candidates:
        return None

    idx = select_movable_index(
        state,
        frozen=frozen,
        moves_cfg=moves_cfg,
        rng=rng,
        exclude_marks=exclude_marks,
    )
    if idx is None:
        return None

    candidate = choose_candidate_with_bias(candidates, candidate_cfg, state.N, rng)
    if candidate is None:
        return None

    old_pos = state.A[idx]
    state.move_mark(idx, candidate)
    return old_pos, candidate


def apply_random_move(
    state: RulerState,
    *,
    frozen: Set[int],
    moves_cfg: Dict[str, Any],
    rng: random.Random,
    exclude_marks: Optional[Set[int]] = None,
) -> Optional[Tuple[int, int]]:
    idx = select_movable_index(
        state,
        frozen=frozen,
        moves_cfg=moves_cfg,
        rng=rng,
        exclude_marks=exclude_marks,
    )
    if idx is None:
        return None

    candidate_cfg = moves_cfg.get("candidate_positions", {})
    occupied = set(state.A)
    lo, hi = 1, state.N - 1

    for _ in range(80):
        if candidate_cfg.get("bias_to_upper_half") and rng.random() < float(candidate_cfg.get("upper_half_prob", 0.75)):
            candidate = rng.randint(max(lo, state.N // 2 + 1), hi)
        else:
            candidate = rng.randint(lo, hi)
        if candidate in occupied:
            continue
        old_pos = state.A[idx]
        state.move_mark(idx, candidate)
        return old_pos, candidate
    return None


def _tail_missing_distances(W: Sequence[int], N: int, tail_window_k: int) -> List[int]:
    lo = max(1, N - int(tail_window_k))
    hi = max(lo, N - 1)
    return [d for d in range(lo, hi + 1) if W[d] == 0]


def build_endpoint_window_candidates(
    state: RulerState,
    *,
    tail_missing: Sequence[int],
    tail_window_k: int,
    low_anchor_max: int,
) -> List[int]:
    occupied = set(state.A)
    seen: Set[int] = set()
    candidates: List[int] = []

    def add(candidate: int) -> None:
        if candidate <= 0 or candidate >= state.N:
            return
        if candidate in occupied or candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    for d in tail_missing:
        add(state.N - int(d))

    lo_tail = max(1, state.N - int(tail_window_k))
    for candidate in range(lo_tail, state.N):
        add(candidate)

    hi_low = min(int(low_anchor_max), state.N - 1)
    for candidate in range(1, hi_low + 1):
        add(candidate)

    return candidates


def apply_endpoint_window_move(
    state: RulerState,
    *,
    frozen: Set[int],
    moves_cfg: Dict[str, Any],
    tail_missing: Sequence[int],
    tail_window_k: int,
    low_anchor_max: int,
    rng: random.Random,
    exclude_marks: Optional[Set[int]] = None,
) -> Optional[Tuple[int, int]]:
    idx = select_movable_index(
        state,
        frozen=frozen,
        moves_cfg=moves_cfg,
        rng=rng,
        exclude_marks=exclude_marks,
    )
    if idx is None:
        return None

    candidates = build_endpoint_window_candidates(
        state,
        tail_missing=tail_missing,
        tail_window_k=tail_window_k,
        low_anchor_max=low_anchor_max,
    )
    if not candidates:
        return None

    candidate_cfg = moves_cfg.get("candidate_positions", {})
    candidate = choose_candidate_with_bias(candidates, candidate_cfg, state.N, rng)
    if candidate is None:
        return None

    old_pos = state.A[idx]
    state.move_mark(idx, candidate)
    return old_pos, candidate


def apply_endpoint_injection_move(
    state: RulerState,
    *,
    frozen: Set[int],
    moves_cfg: Dict[str, Any],
    endpoint_positions: Sequence[int],
    rng: random.Random,
    exclude_marks: Optional[Set[int]] = None,
) -> Optional[Tuple[int, int]]:
    occupied = set(state.A)
    available = [p for p in endpoint_positions if 1 <= p < state.N and p not in occupied]
    if not available:
        return None

    idx = select_movable_index(
        state,
        frozen=frozen,
        moves_cfg=moves_cfg,
        rng=rng,
        exclude_marks=exclude_marks,
    )
    if idx is None:
        return None

    old_pos = state.A[idx]
    new_pos = rng.choice(available)
    state.move_mark(idx, new_pos)
    return old_pos, new_pos


def revert_moves(state: RulerState, moves: Sequence[Tuple[int, int]]) -> None:
    for old_pos, new_pos in reversed(moves):
        idx = state.A.index(new_pos)
        state.move_mark(idx, old_pos)


def verify_complete_independent(marks: Sequence[int], N: int) -> Tuple[bool, List[int]]:
    W = compute_W(marks, N)
    missing = [d for d in range(1, N + 1) if W[d] == 0]
    return len(missing) == 0, missing


def artifact_integrity(marks: Sequence[int], N: int, m_try: int) -> Dict[str, bool]:
    return {
        "sorted": list(marks) == sorted(marks),
        "unique": len(set(marks)) == len(marks),
        "endpoints_fixed": bool(marks) and marks[0] == 0 and marks[-1] == N,
        "len_ok": len(marks) == m_try,
    }


def _assert_integrity(checks: Dict[str, bool], seed: int, N: int, m_try: int) -> None:
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        joined = ", ".join(failed)
        raise ValueError(
            f"artifact integrity failed for seed={seed} N={N} m_try={m_try}: {joined}"
        )


def _write_seed_payload(seed_dir: Path, payload: SeedOutcome, trace: Sequence[Dict[str, Any]], save_trace: bool) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "best.json").write_text(json.dumps(payload.__dict__, indent=2))
    if save_trace:
        trace_path = seed_dir / "trace.jsonl"
        with trace_path.open("w") as handle:
            for row in trace:
                handle.write(json.dumps(row))
                handle.write("\n")


def _seed_values(base_seed: int, target_slot: int, stage_slot: int, candidate_slot: int, count: int) -> List[int]:
    values: List[int] = []
    root = base_seed + target_slot * 1_000_000 + stage_slot * 100_000 + candidate_slot * 1_000
    for i in range(count):
        values.append(root + i)
    return values


def run_seed(
    *,
    N: int,
    m_try: int,
    initial_marks: Sequence[int],
    seed: int,
    iterations: int,
    freeze_policy: Dict[str, Any],
    objective_cfg: Dict[str, Any],
    moves_cfg: Dict[str, Any],
    coupled_cfg: Dict[str, Any],
    endpoint_cfg: Dict[str, Any],
    endpoint_window_cfg: Dict[str, Any],
    log_interval: int,
    save_best_every: int,
    seed_dir: Optional[Path],
    save_trace: bool,
) -> SeedOutcome:
    rng = random.Random(seed)
    state = RulerState(N, initial_marks)
    if seed_dir is not None:
        seed_dir.mkdir(parents=True, exist_ok=True)

    current_score = compute_objective_score(state, objective_cfg)
    best_state = RulerState(N, state.A)
    best_score = current_score
    best_missing = missing_distances_from_W(best_state.W, N)
    best_step = 0

    coupled_attempts = 0
    coupled_accepts = 0
    endpoint_window_attempts = 0
    endpoint_window_accepts = 0
    tail_trigger_count = 0
    endpoint_fired = False

    endpoint_targets = [int(v) for v in endpoint_cfg.get("target_distances", [N - 2])]
    endpoint_streak_threshold = int(endpoint_cfg.get("streak_steps", 5000))
    endpoint_positions = [int(v) for v in endpoint_cfg.get("attempt_positions", [2, N - 1, N - 2])]
    endpoint_streaks = {d: 0 for d in endpoint_targets}

    trace: List[Dict[str, Any]] = []

    repair_prob = float(moves_cfg.get("repair_prob", 0.9))
    equal_accept_prob = float(moves_cfg.get("accept_equal_prob", 0.02))
    endpoint_window_enabled = bool(endpoint_window_cfg.get("enabled", False))
    endpoint_window_best_missing_max = int(endpoint_window_cfg.get("trigger_best_missing_at_most", 12))
    endpoint_window_stall_steps = int(endpoint_window_cfg.get("trigger_stall_steps", 3000))
    endpoint_window_tail_window_k = int(endpoint_window_cfg.get("tail_window_k", 16))
    endpoint_window_low_anchor_max = int(endpoint_window_cfg.get("low_anchor_max", 12))
    endpoint_window_prob = float(endpoint_window_cfg.get("proposal_prob", 0.45))

    for step in range(1, iterations + 1):
        for d in endpoint_streaks:
            if 1 <= d <= N and state.W[d] == 0:
                endpoint_streaks[d] += 1
            else:
                endpoint_streaks[d] = 0

        frozen = build_frozen_set(state, freeze_policy)

        trigger = coupled_cfg.get("trigger", {})
        stall_steps = int(trigger.get("stall_steps", 5000))
        best_missing_limit = int(trigger.get("best_missing_count_at_most", 25))
        coupled_prob = float(coupled_cfg.get("proposal_prob", 0.35))

        use_coupled_default = (
            coupled_cfg.get("enabled", True)
            and (step - best_step) >= stall_steps
            and best_score.missing_count <= best_missing_limit
        )
        use_coupled = use_coupled_default and rng.random() < coupled_prob

        tail_missing = _tail_missing_distances(state.W, N, endpoint_window_tail_window_k)
        endpoint_window_trigger_base = (
            endpoint_window_enabled
            and (step - best_step) >= endpoint_window_stall_steps
            and best_score.missing_count <= endpoint_window_best_missing_max
            and len(tail_missing) > 0
        )
        if endpoint_window_trigger_base:
            tail_trigger_count += 1

        use_endpoint_window = endpoint_window_trigger_base and rng.random() < endpoint_window_prob
        if use_endpoint_window:
            use_coupled = True

        accepted = False

        if use_coupled:
            coupled_attempts += 1
            old_score = current_score
            moves: List[Tuple[int, int]] = []
            endpoint_window_used = False

            if use_endpoint_window:
                endpoint_window_attempts += 1
                first = apply_endpoint_window_move(
                    state,
                    frozen=frozen,
                    moves_cfg=moves_cfg,
                    tail_missing=tail_missing,
                    tail_window_k=endpoint_window_tail_window_k,
                    low_anchor_max=endpoint_window_low_anchor_max,
                    rng=rng,
                )
                if first is not None:
                    moves.append(first)
                    endpoint_window_used = True

            if not moves:
                endpoint_triggered = bool(endpoint_cfg.get("enabled", True)) and any(
                    streak >= endpoint_streak_threshold for streak in endpoint_streaks.values()
                )
                if endpoint_triggered:
                    first = apply_endpoint_injection_move(
                        state,
                        frozen=frozen,
                        moves_cfg=moves_cfg,
                        endpoint_positions=endpoint_positions,
                        rng=rng,
                    )
                    if first is not None:
                        moves.append(first)
                        endpoint_fired = True
                else:
                    if rng.random() < repair_prob:
                        first = apply_repair_move(
                            state,
                            frozen=frozen,
                            moves_cfg=moves_cfg,
                            objective_cfg=objective_cfg,
                            rng=rng,
                        )
                    else:
                        first = apply_random_move(
                            state,
                            frozen=frozen,
                            moves_cfg=moves_cfg,
                            rng=rng,
                        )
                    if first is not None:
                        moves.append(first)

            if moves:
                moved_marks = {moves[0][1]}
                frozen2 = build_frozen_set(state, freeze_policy)
                second = apply_repair_move(
                    state,
                    frozen=frozen2,
                    moves_cfg=moves_cfg,
                    objective_cfg=objective_cfg,
                    rng=rng,
                    exclude_marks=moved_marks,
                )
                if second is None:
                    second = apply_random_move(
                        state,
                        frozen=frozen2,
                        moves_cfg=moves_cfg,
                        rng=rng,
                        exclude_marks=moved_marks,
                    )
                if second is not None:
                    moves.append(second)

            if len(moves) == 2:
                new_score = compute_objective_score(state, objective_cfg)
                if new_score.key() < old_score.key():
                    accepted = True
                    current_score = new_score
                    coupled_accepts += 1
                    if endpoint_window_used:
                        endpoint_window_accepts += 1
                else:
                    revert_moves(state, moves)
            elif moves:
                revert_moves(state, moves)

        else:
            old_score = current_score
            if rng.random() < repair_prob:
                move = apply_repair_move(
                    state,
                    frozen=frozen,
                    moves_cfg=moves_cfg,
                    objective_cfg=objective_cfg,
                    rng=rng,
                )
            else:
                move = apply_random_move(
                    state,
                    frozen=frozen,
                    moves_cfg=moves_cfg,
                    rng=rng,
                )

            if move is not None:
                new_score = compute_objective_score(state, objective_cfg)
                if new_score.key() < old_score.key() or (
                    new_score.key() == old_score.key() and rng.random() < equal_accept_prob
                ):
                    accepted = True
                    current_score = new_score
                else:
                    revert_moves(state, [move])

        if accepted and current_score.key() < best_score.key():
            best_state = RulerState(N, state.A)
            best_score = current_score
            best_missing = missing_distances_from_W(best_state.W, N)
            best_step = step

        if log_interval > 0 and step % log_interval == 0:
            trace.append(
                {
                    "step": step,
                    "current_missing_count": state.missing_count,
                    "current_M25": missing_count_upto(state.W, 25),
                    "current_M50": missing_count_upto(state.W, 50),
                    "best_missing_count": best_score.missing_count,
                    "best_secondary": best_score.secondary,
                    "coupled_attempts": coupled_attempts,
                    "coupled_accepts": coupled_accepts,
                    "endpoint_window_attempts": endpoint_window_attempts,
                    "endpoint_window_accepts": endpoint_window_accepts,
                    "tail_trigger_count": tail_trigger_count,
                    "endpoint_hole_mode_fired": 1 if endpoint_fired else 0,
                }
            )

        if save_best_every > 0 and seed_dir is not None and step % save_best_every == 0:
            checkpoint = {
                "step": step,
                "best_missing_count": best_score.missing_count,
                "best_secondary": best_score.secondary,
                "best_M25": best_score.M25,
                "best_M50": best_score.M50,
                "endpoint_window_attempts": endpoint_window_attempts,
                "endpoint_window_accepts": endpoint_window_accepts,
                "tail_trigger_count": tail_trigger_count,
            }
            (seed_dir / "best_checkpoint.json").write_text(json.dumps(checkpoint, indent=2))

    complete_ok, independent_missing = verify_complete_independent(best_state.A, N)
    if best_score.missing_count != len(independent_missing):
        raise ValueError(
            "state cache mismatch: tracked missing_count differs from independent verification "
            f"(tracked={best_score.missing_count}, independent={len(independent_missing)})"
        )
    checks = artifact_integrity(best_state.A, N, m_try)
    _assert_integrity(checks, seed, N, m_try)

    outcome = SeedOutcome(
        seed=seed,
        iterations_run=iterations,
        objective_key=best_score.key(),
        best_missing_count=best_score.missing_count,
        best_secondary=best_score.secondary,
        M25=best_score.M25,
        M50=best_score.M50,
        best_marks=list(best_state.A),
        best_missing_list=independent_missing,
        coupled_attempts=coupled_attempts,
        coupled_accepts=coupled_accepts,
        endpoint_window_attempts=endpoint_window_attempts,
        endpoint_window_accepts=endpoint_window_accepts,
        tail_trigger_count=tail_trigger_count,
        endpoint_hole_mode_fired=endpoint_fired,
        complete_independent=complete_ok,
        integrity_checks=checks,
    )

    if seed_dir is not None:
        _write_seed_payload(seed_dir, outcome, trace, save_trace)

    return outcome


def generate_deletion_candidates(start_marks: Sequence[int], N: int) -> List[Dict[str, Any]]:
    marks = sorted(start_marks)
    if not marks:
        raise ValueError("start_marks cannot be empty")
    if marks[0] != 0 or marks[-1] != N:
        raise ValueError("start_marks must include endpoints 0 and N")
    if len(set(marks)) != len(marks):
        raise ValueError("start_marks must be unique")

    candidates: List[Dict[str, Any]] = []
    for idx in range(1, len(marks) - 1):
        deleted_mark = marks[idx]
        reduced = marks[:idx] + marks[idx + 1 :]
        candidates.append(
            {
                "id": f"delete_{deleted_mark}_idx_{idx}",
                "deleted_index": idx,
                "deleted_mark": deleted_mark,
                "initial_marks": reduced,
            }
        )
    return candidates


def _candidate_sort_key(row: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    key = tuple(row["stage_best_objective_key"])
    return (*key, int(row["deleted_mark"]))


def _classify_holes(missing: Sequence[int], N: int) -> Dict[str, int]:
    counts = {"endpoint": 0, "mid_scaffold": 0, "other": 0}
    lo_mid = int(0.45 * N)
    hi_mid = int(0.75 * N)
    for d in missing:
        if d <= 2 or d >= N - 2:
            counts["endpoint"] += 1
        elif lo_mid <= d <= hi_mid:
            counts["mid_scaffold"] += 1
        else:
            counts["other"] += 1
    return counts


def _default_config() -> Dict[str, Any]:
    return {
        "suite": "E_hunt_delete_and_repair",
        "random_seed_base": 20260208,
        "freeze_policy": {
            "always_freeze": [0, "N"],
            "freeze_prefix_le": 8,
        },
        "objective": {
            "missing_power_p": 2,
            "kernel_penalty_lambda": 2000,
            "kernel_guard": {
                "require_M25": 0,
                "allow_M50_max": 2,
            },
        },
        "moves": {
            "repair_prob": 0.90,
            "random_prob": 0.10,
            "accept_equal_prob": 0.02,
            "anchors": {
                "le_value": 120,
                "top_high": 6,
            },
            "candidate_positions": {
                "pool_size": 120,
                "dedupe": True,
                "bias_to_upper_half": True,
                "upper_half_prob": 0.75,
            },
            "mark_selection": {
                "top_k_bias": 8,
                "responsibility_max_d": 50,
            },
            "coupled": {
                "enabled": True,
                "trigger": {
                    "stall_steps": 5000,
                    "best_missing_count_at_most": 25,
                },
                "proposal_prob": 0.35,
            },
            "endpoint_hole_mode": {
                "enabled": True,
                "target_distances": ["N-2"],
                "attempt_positions": [2, "N-1", "N-2"],
                "streak_steps": 5000,
            },
            "endpoint_window_mode": {
                "enabled": False,
                "trigger_best_missing_at_most": 12,
                "trigger_stall_steps": 3000,
                "tail_window_k": 16,
                "low_anchor_max": 12,
                "proposal_prob": 0.45,
            },
        },
        "search": {
            "log_interval": 1000,
            "save_best_every": 5000,
            "skip_championship_if_complete": True,
            "stop_on_first_complete_global": False,
            "resume_existing_candidates": True,
            "stages": {
                "screen": {
                    "enabled": True,
                    "seeds": 6,
                    "iterations_per_seed": 30000,
                    "promote_top_k": 6,
                },
                "deep": {
                    "enabled": True,
                    "seeds": 24,
                    "iterations_per_seed": 160000,
                    "promote_top_k": 2,
                },
                "championship": {
                    "enabled": True,
                    "seeds": 50,
                    "iterations_per_seed": 250000,
                },
            },
        },
        "outputs": {
            "root_dir": "results/e_hunt_delete_repair",
            "save_trace": True,
        },
    }


def load_e_hunt_config(config_path: Path) -> Dict[str, Any]:
    cfg = json.loads(config_path.read_text())
    merged = _merge_dict(_default_config(), cfg)

    if "targets" not in merged or not merged["targets"]:
        raise ValueError("config must define non-empty targets")

    return merged


def run_e_hunt(config: Dict[str, Any]) -> Dict[str, Any]:
    root_dir = Path(config["outputs"]["root_dir"])
    root_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(config.get("random_seed_base", 20260208))

    freeze_policy = dict(config["freeze_policy"])
    objective_cfg = dict(config["objective"])
    moves_cfg = dict(config["moves"])
    coupled_cfg = dict(moves_cfg.get("coupled", {}))
    endpoint_cfg_template = dict(moves_cfg.get("endpoint_hole_mode", {}))
    endpoint_window_cfg = dict(moves_cfg.get("endpoint_window_mode", {}))

    search_cfg = dict(config["search"])
    stages_cfg = dict(search_cfg["stages"])
    log_interval = int(search_cfg.get("log_interval", 1000))
    save_best_every = int(search_cfg.get("save_best_every", 5000))
    skip_champ_if_complete = bool(search_cfg.get("skip_championship_if_complete", True))
    stop_on_first_complete_global = bool(search_cfg.get("stop_on_first_complete_global", False))
    resume_existing_candidates = bool(search_cfg.get("resume_existing_candidates", True))

    overall_targets: List[Dict[str, Any]] = []
    total_complete_hits = 0
    stopped_early_global = False

    for target_slot, target in enumerate(config["targets"]):
        if stopped_early_global:
            break
        N = int(target["N"])
        start_marks = [int(v) for v in target["start_marks"]]
        m_try = int(target["m_try"])
        if len(start_marks) - 1 != m_try:
            raise ValueError(f"target N={N} expected m_try=len(start_marks)-1")

        endpoint_cfg = _json_clone(endpoint_cfg_template)
        endpoint_cfg["target_distances"] = _resolve_int_list(endpoint_cfg.get("target_distances", ["N-2"]), N)
        endpoint_cfg["attempt_positions"] = _resolve_int_list(endpoint_cfg.get("attempt_positions", [2, "N-1", "N-2"]), N)

        target_slug = f"N{N}_m{m_try}"
        target_dir = root_dir / target_slug
        target_dir.mkdir(parents=True, exist_ok=True)

        candidates = generate_deletion_candidates(start_marks, N)
        allowlist = target.get("candidate_allowlist_deleted_marks")
        if allowlist is not None:
            allowed_deleted_marks = {int(value) for value in allowlist}
            candidates = [candidate for candidate in candidates if int(candidate["deleted_mark"]) in allowed_deleted_marks]
            if not candidates:
                raise ValueError(f"target N={N} candidate_allowlist_deleted_marks filtered out all candidates")
        candidate_lookup = {c["id"]: c for c in candidates}

        stage_summaries: List[Dict[str, Any]] = []
        all_candidate_rows: List[Dict[str, Any]] = []
        active_ids = [c["id"] for c in candidates]
        has_complete = False

        for stage_slot, stage_name in enumerate(STAGE_ORDER):
            stage_cfg = stages_cfg.get(stage_name, {})
            if not stage_cfg.get("enabled", False):
                continue
            if stage_name == "championship" and skip_champ_if_complete and has_complete:
                continue
            if not active_ids:
                continue

            stage_candidate_ids = list(active_ids)
            max_candidates = stage_cfg.get("max_candidates")
            if max_candidates is not None:
                stage_candidate_ids = stage_candidate_ids[: int(max_candidates)]

            seeds_count = int(stage_cfg["seeds"])
            iterations = int(stage_cfg["iterations_per_seed"])

            candidate_rows: List[Dict[str, Any]] = []
            for candidate_slot, candidate_id in enumerate(stage_candidate_ids):
                candidate = candidate_lookup[candidate_id]
                candidate_dir = target_dir / f"stage_{stage_name}" / candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                candidate_summary_path = candidate_dir / "candidate_summary.json"

                if resume_existing_candidates and candidate_summary_path.exists():
                    existing = json.loads(candidate_summary_path.read_text())
                    is_compatible = (
                        existing.get("stage") == stage_name
                        and int(existing.get("N", -1)) == N
                        and int(existing.get("m_try", -1)) == m_try
                        and existing.get("candidate_id") == candidate_id
                        and len(existing.get("evaluated_seeds", [])) == seeds_count
                    )
                    if is_compatible:
                        candidate_rows.append(existing)
                        all_candidate_rows.append(existing)
                        if (
                            existing.get("stage_best_complete_independent")
                            and int(existing.get("stage_best_missing_count", 1)) == 0
                        ):
                            has_complete = True
                            if stop_on_first_complete_global and existing.get("rerun_verified") in {True, None}:
                                stopped_early_global = True
                        if stopped_early_global:
                            break
                        continue

                seed_values = _seed_values(base_seed, target_slot, stage_slot, candidate_slot, seeds_count)
                per_seed: List[SeedOutcome] = []
                holes_counter: Counter[int] = Counter()

                for seed in seed_values:
                    seed_dir = candidate_dir / f"seed_{seed}"
                    outcome = run_seed(
                        N=N,
                        m_try=m_try,
                        initial_marks=candidate["initial_marks"],
                        seed=seed,
                        iterations=iterations,
                        freeze_policy=freeze_policy,
                        objective_cfg=objective_cfg,
                        moves_cfg=moves_cfg,
                        coupled_cfg=coupled_cfg,
                        endpoint_cfg=endpoint_cfg,
                        endpoint_window_cfg=endpoint_window_cfg,
                        log_interval=log_interval,
                        save_best_every=save_best_every,
                        seed_dir=seed_dir,
                        save_trace=bool(config["outputs"].get("save_trace", True)),
                    )
                    per_seed.append(outcome)
                    holes_counter.update(outcome.best_missing_list)

                best_seed_outcome = min(per_seed, key=lambda row: row.objective_key)
                complete_hits = sum(1 for row in per_seed if row.complete_independent and row.best_missing_count == 0)

                rerun_verified: Optional[bool] = None
                if best_seed_outcome.complete_independent and best_seed_outcome.best_missing_count == 0:
                    rerun = run_seed(
                        N=N,
                        m_try=m_try,
                        initial_marks=candidate["initial_marks"],
                        seed=best_seed_outcome.seed,
                        iterations=iterations,
                        freeze_policy=freeze_policy,
                        objective_cfg=objective_cfg,
                        moves_cfg=moves_cfg,
                        coupled_cfg=coupled_cfg,
                        endpoint_cfg=endpoint_cfg,
                        endpoint_window_cfg=endpoint_window_cfg,
                        log_interval=0,
                        save_best_every=0,
                        seed_dir=None,
                        save_trace=False,
                    )
                    rerun_verified = (
                        rerun.complete_independent
                        and rerun.best_missing_count == 0
                        and rerun.best_marks == best_seed_outcome.best_marks
                    )

                row = {
                    "stage": stage_name,
                    "N": N,
                    "m_try": m_try,
                    "deleted_index": candidate["deleted_index"],
                    "deleted_mark": candidate["deleted_mark"],
                    "candidate_id": candidate_id,
                    "evaluated_seeds": seed_values,
                    "stage_best_seed": best_seed_outcome.seed,
                    "stage_best_objective_key": list(best_seed_outcome.objective_key),
                    "stage_best_missing_count": best_seed_outcome.best_missing_count,
                    "stage_best_secondary": best_seed_outcome.best_secondary,
                    "stage_best_M25": best_seed_outcome.M25,
                    "stage_best_M50": best_seed_outcome.M50,
                    "stage_best_marks": best_seed_outcome.best_marks,
                    "stage_best_missing_list": best_seed_outcome.best_missing_list,
                    "stage_best_complete_independent": best_seed_outcome.complete_independent,
                    "complete_hits": complete_hits,
                    "coupled_attempts": best_seed_outcome.coupled_attempts,
                    "coupled_accepts": best_seed_outcome.coupled_accepts,
                    "endpoint_window_attempts": best_seed_outcome.endpoint_window_attempts,
                    "endpoint_window_accepts": best_seed_outcome.endpoint_window_accepts,
                    "tail_trigger_count": best_seed_outcome.tail_trigger_count,
                    "endpoint_hole_mode_fired": best_seed_outcome.endpoint_hole_mode_fired,
                    "rerun_verified": rerun_verified,
                    "holes_top10": holes_counter.most_common(10),
                    "artifact_dir": str(candidate_dir),
                }
                candidate_rows.append(row)
                all_candidate_rows.append(row)

                if row["stage_best_complete_independent"] and row["stage_best_missing_count"] == 0:
                    has_complete = True
                    if stop_on_first_complete_global and (row["rerun_verified"] in {True, None}):
                        stopped_early_global = True

                candidate_summary_path.write_text(json.dumps(row, indent=2))

                if stopped_early_global:
                    break

            candidate_rows.sort(key=_candidate_sort_key)

            stage_summary = {
                "name": stage_name,
                "iterations_per_seed": iterations,
                "seeds_per_candidate": seeds_count,
                "evaluated_candidates": len(candidate_rows),
                "complete_hits": sum(r["complete_hits"] for r in candidate_rows),
                "rows": candidate_rows,
            }
            stage_summaries.append(stage_summary)

            if stopped_early_global:
                break

            if stage_name != "championship":
                promote_top_k = int(stage_cfg.get("promote_top_k", len(candidate_rows)))
                active_ids = [row["candidate_id"] for row in candidate_rows[:promote_top_k]]

        best_overall = min(all_candidate_rows, key=_candidate_sort_key)
        best_hole_taxonomy = _classify_holes(best_overall["stage_best_missing_list"], N)

        target_complete = (
            best_overall["stage_best_complete_independent"]
            and best_overall["stage_best_missing_count"] == 0
            and (best_overall["rerun_verified"] in {True, None})
        )
        if target_complete:
            total_complete_hits += 1
            if stop_on_first_complete_global:
                stopped_early_global = True

        target_summary = {
            "N": N,
            "m_try": m_try,
            "start_marks": start_marks,
            "deletion_candidates_count": len(candidates),
            "candidate_allowlist_deleted_marks": None if allowlist is None else [int(v) for v in allowlist],
            "stages": stage_summaries,
            "best_overall": best_overall,
            "best_hole_taxonomy": best_hole_taxonomy,
            "complete_target": target_complete,
        }
        overall_targets.append(target_summary)

        (target_dir / "target_summary.json").write_text(json.dumps(target_summary, indent=2))

    summary = {
        "suite": config.get("suite", "E_hunt_delete_and_repair"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "targets": overall_targets,
        "overall": {
            "target_count": len(overall_targets),
            "total_complete_hits": total_complete_hits,
            "success_targets": [t["N"] for t in overall_targets if t["complete_target"]],
            "stopped_early_global": stopped_early_global,
        },
    }

    summary_path = root_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


def run_e_hunt_from_config_path(config_path: Path) -> Dict[str, Any]:
    config = load_e_hunt_config(config_path)
    return run_e_hunt(config)
