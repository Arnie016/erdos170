from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .metrics import (
    ImpossibleResult,
    MetricSummary,
    Solution,
    compute_W,
    energy_floor,
    format_csv_header,
    summarize_metrics,
)

ALPHA = 1_000_000
BETA = 1
GAMMA = 1000


@dataclass
class AnnealResult:
    A: List[int]
    metrics: MetricSummary
    H: int
    kernel_A: Optional[List[int]] = None
    kernel_M50: Optional[int] = None
    ladder_proposals: int = 0
    ladder_accepts: int = 0
    best_kernel_seen: Optional[int] = None


def initial_ruler(N: int, m: int, rng: random.Random) -> List[int]:
    if m < 2:
        raise ValueError("m must be >= 2")
    if m == 2:
        return [0, N]
    positions = {0, N}
    A = [0, N]
    if N >= 200 and m >= 3:
        positions.add(1)
        A.append(1)
    if N >= 200 and m >= 4:
        positions.add(2)
        A.append(2)
    k = 1
    while len(A) < m:
        pos = round(k * N / (m - 1))
        pos = max(1, min(N - 1, pos))
        while pos in positions:
            pos = pos + 1
            if pos >= N:
                pos = 1
        positions.add(pos)
        A.append(pos)
        k += 1
    return sorted(A)


class RulerState:
    def __init__(self, N: int, A: Sequence[int]):
        self.N = N
        self.A = list(sorted(A))
        self.m = len(self.A)
        self.W = compute_W(self.A, N)
        self.S = self.m * (self.m - 1) // 2
        self.missing_count = sum(1 for d in range(1, N + 1) if self.W[d] == 0)
        self.E = sum(w * w for w in self.W[1:])
        self.peak = max(self.W[1:]) if N > 0 else 0

    def hamiltonian(self) -> int:
        return ALPHA * self.missing_count + BETA * self.E + GAMMA * self.peak

    def move_mark(self, idx: int, new_pos: int) -> Tuple[int, int, bool]:
        old_pos = self.A[idx]
        if old_pos == new_pos:
            return old_pos, new_pos, False
        needs_peak_recalc = False
        # remove old contributions
        for mark in self.A:
            if mark == old_pos:
                continue
            d = abs(old_pos - mark)
            old_w = self.W[d]
            self.W[d] = old_w - 1
            self.E += (old_w - 1) * (old_w - 1) - old_w * old_w
            if old_w == 1:
                self.missing_count += 1
            if old_w == self.peak and self.W[d] < self.peak:
                needs_peak_recalc = True
        # add new contributions
        for mark in self.A:
            if mark == old_pos:
                continue
            d = abs(new_pos - mark)
            old_w = self.W[d]
            self.W[d] = old_w + 1
            self.E += (old_w + 1) * (old_w + 1) - old_w * old_w
            if old_w == 0:
                self.missing_count -= 1
            if self.W[d] > self.peak:
                self.peak = self.W[d]
        # update positions
        self.A.pop(idx)
        insert_at = 0
        while insert_at < len(self.A) and self.A[insert_at] < new_pos:
            insert_at += 1
        self.A.insert(insert_at, new_pos)
        if needs_peak_recalc:
            self.peak = max(self.W[1:]) if self.N > 0 else 0
        return old_pos, new_pos, True


def anneal_run(
    N: int,
    m: int,
    steps: int,
    rng: random.Random,
    temperature: float = 5000.0,
    cooling: float = 0.9995,
    stop_rho: float = 1.05,
    phase1_ratio: float = 0.6,
) -> AnnealResult:
    A = initial_ruler(N, m, rng)
    state = RulerState(N, A)
    best_state = RulerState(N, state.A)
    best_h = best_state.hamiltonian()
    phase1_steps = max(1, int(steps * phase1_ratio))
    phase2_steps = max(1, steps - phase1_steps)
    best_complete_state: Optional[RulerState] = None
    best_complete_h = None
    stage0_steps = min(50_000, phase1_steps)
    frozen_targets: set[int] = set()
    freeze_ttl = 0
    ladder_proposals = 0
    ladder_accepts = 0
    best_kernel_state: Optional[RulerState] = None
    best_kernel_missing = None

    def is_frozen(idx: int) -> bool:
        return freeze_ttl > 0 and state.A[idx] in frozen_targets

    def propose_move(targeted_prob: float) -> Tuple[int, int, int, int, int, int, int, bool]:
        idx = rng.randint(1, m - 2)
        if is_frozen(idx):
            return idx, state.A[idx], state.A[idx], state.missing_count, state.peak, state.E, spike_mass_from_W(state.W), False
        occupied = set(state.A)
        proposal = None
        if state.missing_count > 0 and rng.random() < targeted_prob:
            missing = [d for d in range(1, N + 1) if state.W[d] == 0]
            if missing:
                d = rng.choice(missing)
                anchor = rng.choice(state.A)
                if rng.random() < 0.5:
                    candidate = anchor + d
                else:
                    candidate = anchor - d
                if 1 <= candidate <= N - 1 and candidate not in occupied:
                    proposal = candidate
        if proposal is None:
            proposal = rng.randint(1, N - 1)
            while proposal in occupied:
                proposal = rng.randint(1, N - 1)
        old_missing = state.missing_count
        old_peak = state.peak
        old_E = state.E
        old_spike = spike_mass_from_W(state.W)
        old_pos, new_pos, moved = state.move_mark(idx, proposal)
        return idx, old_pos, new_pos, old_missing, old_peak, old_E, old_spike, moved

    def spike_mass_from_W(W: Sequence[int]) -> int:
        return sum(max(w - 2, 0) for w in W[1:])

    def missing_small(K: int) -> List[int]:
        return [d for d in range(1, min(K, N) + 1) if state.W[d] == 0]

    def missing_small_count(K: int) -> int:
        return sum(1 for d in range(1, min(K, N) + 1) if state.W[d] == 0)

    def select_low_loss_mark(K: int) -> int:
        best_idx = None
        best_loss = None
        for idx in range(1, m - 1):
            if is_frozen(idx):
                continue
            loss = 0
            ai = state.A[idx]
            for j, aj in enumerate(state.A):
                if j == idx:
                    continue
                d = abs(ai - aj)
                if d <= K and state.W[d] == 1:
                    loss += 1
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_idx = idx
        return best_idx if best_idx is not None else 1

    def create_spacing_move(K: int) -> Tuple[int, int, int, int, int, int, int, bool]:
        missing = missing_small(K)
        if not missing:
            return 0, 0, 0, 0, 0, 0, 0, False
        weights = [1 / d for d in missing]
        total = sum(weights)
        r = rng.random() * total
        chosen_d = missing[-1]
        for d, w in zip(missing, weights):
            r -= w
            if r <= 0:
                chosen_d = d
                break
        anchor = rng.choice(state.A)
        candidates = [anchor - chosen_d, anchor + chosen_d]
        rng.shuffle(candidates)
        occupied = set(state.A)
        for candidate in candidates:
            if 1 <= candidate <= N - 1 and candidate not in occupied:
                idx = select_low_loss_mark(K)
                if is_frozen(idx):
                    continue
                old_missing = state.missing_count
                old_peak = state.peak
                old_E = state.E
                old_spike = spike_mass_from_W(state.W)
                old_pos, new_pos, moved = state.move_mark(idx, candidate)
                return idx, old_pos, new_pos, old_missing, old_peak, old_E, old_spike, moved
        return 0, 0, 0, 0, 0, 0, 0, False

    def two_mark_move(K: int) -> Tuple[int, int, int, int, int, int, int, bool]:
        missing = missing_small(K)
        if len(missing) < 2:
            return 0, 0, 0, 0, 0, 0, 0, False
        first = rng.choice(missing)
        second = rng.choice([d for d in missing if d != first])
        anchors = [rng.choice(state.A), rng.choice(state.A)]
        targets = [anchors[0] + first, anchors[0] - first, anchors[1] + second, anchors[1] - second]
        rng.shuffle(targets)
        occupied = set(state.A)
        idxs = list(range(1, m - 1))
        rng.shuffle(idxs)
        moves = []
        for idx, target in zip(idxs[:2], targets[:2]):
            if 1 <= target <= N - 1 and target not in occupied:
                moves.append((idx, target))
                occupied.add(target)
        if len(moves) != 2:
            return 0, 0, 0, 0, 0, 0, 0, False
        old_missing = state.missing_count
        old_peak = state.peak
        old_E = state.E
        old_spike = spike_mass_from_W(state.W)
        (idx1, target1), (idx2, target2) = moves
        old_pos1 = state.A[idx1]
        old_pos2 = state.A[idx2]
        for idx, target in sorted([(idx1, target1), (idx2, target2)], reverse=True):
            state.move_mark(idx, target)
        return idx2, old_pos2, target2, old_missing, old_peak, old_E, old_spike, True

    def residue_penalty(q: int) -> int:
        counts = [0] * q
        for mark in state.A:
            counts[mark % q] += 1
        return sum(count * count for count in counts)

    def is_valid_A(A: Sequence[int]) -> bool:
        if not A or A[0] != 0 or A[-1] != N:
            return False
        return all(a < b for a, b in zip(A, A[1:]))

    def ladder_injection_move(K: int) -> Tuple[List[Tuple[int, int, int]], int, int, int, int, bool, List[int]]:
        base = rng.randint(0, min(20, N - 6))
        targets = [base + offset for offset in range(6)]
        occupied = set(state.A)
        movable = [idx for idx in range(1, m - 1) if state.A[idx] not in targets]
        if len(movable) < 2:
            return [], 0, 0, 0, 0, False, []
        penalties: List[Tuple[int, int]] = []
        for idx in movable:
            mark = state.A[idx]
            penalty = 0
            for other in state.A:
                if other == mark:
                    continue
                d = abs(mark - other)
                if state.W[d] == 1:
                    penalty += 1
            penalties.append((penalty, idx))
        penalties.sort()
        moves: List[Tuple[int, int, int]] = []
        used_targets = set()
        for target, (_penalty, idx) in zip(targets, penalties):
            if target in occupied or target in used_targets:
                continue
            moves.append((idx, state.A[idx], target))
            used_targets.add(target)
        if len(moves) < 4:
            return [], 0, 0, 0, 0, False, []
        old_missing = state.missing_count
        old_peak = state.peak
        old_E = state.E
        old_spike = spike_mass_from_W(state.W)
        for idx, _old_pos, new_pos in sorted(moves, reverse=True):
            state.move_mark(idx, new_pos)
        return moves, old_missing, old_peak, old_E, old_spike, True, targets

    def forced_ladder_injection() -> Tuple[List[Tuple[int, int, int]], List[int]]:
        base = rng.randint(0, min(20, N - 6))
        targets = [base + offset for offset in range(6)]
        occupied = set(state.A)
        candidates = [idx for idx in range(1, m - 1) if state.A[idx] % 8 == 0 and state.A[idx] not in targets]
        if len(candidates) < 6:
            candidates = [idx for idx in range(1, m - 1) if state.A[idx] not in targets]
        rng.shuffle(candidates)
        moves: List[Tuple[int, int, int]] = []
        for idx, target in zip(candidates[:6], targets):
            if target in occupied:
                continue
            moves.append((idx, state.A[idx], target))
            occupied.add(target)
        if len(moves) < 4:
            return [], []
        for idx, _old_pos, new_pos in sorted(moves, reverse=True):
            state.move_mark(idx, new_pos)
        return moves, targets

    def gap_from_state() -> int:
        Emin = energy_floor(state.S, N)
        return state.E - Emin

    def try_spike_move() -> Tuple[int, int, int, int, int, int, int, bool]:
        defect_distances = [d for d in range(1, N + 1) if state.W[d] >= 4]
        if not defect_distances:
            defect_distances = [d for d in range(1, N + 1) if state.W[d] >= 3]
        if not defect_distances:
            return 0, 0, 0, 0, 0, 0, 0, False
        target_d = rng.choice(defect_distances)
        indices = list(range(1, m - 1))
        rng.shuffle(indices)
        occupied = set(state.A)
        for i in indices:
            ai = state.A[i]
            for j in range(m):
                if j == i:
                    continue
                aj = state.A[j]
                if abs(ai - aj) != target_d:
                    continue
                deltas = [-3, -2, -1, 1, 2, 3]
                rng.shuffle(deltas)
                for delta in deltas:
                    candidate = ai + delta
                    if not (1 <= candidate <= N - 1):
                        continue
                    if candidate in occupied:
                        continue
                    old_missing = state.missing_count
                    old_peak = state.peak
                    old_E = state.E
                    old_spike = spike_mass_from_W(state.W)
                    old_pos, new_pos, moved = state.move_mark(i, candidate)
                    return i, old_pos, new_pos, old_missing, old_peak, old_E, old_spike, moved
        return 0, 0, 0, 0, 0, 0, 0, False

    # Phase 1: complete-first
    for step in range(phase1_steps):
        if freeze_ttl > 0:
            freeze_ttl -= 1
            if freeze_ttl == 0:
                frozen_targets.clear()
        if state.missing_count == 0:
            best_complete_state = RulerState(N, state.A)
            best_complete_h = ALPHA * 0 + GAMMA * state.peak
            break
        K = 50 if N >= 200 else 0
        missing_kernel = missing_small_count(K) if K else 0
        residue_score = residue_penalty(8) if K else 0
        if K:
            if is_valid_A(state.A) and (best_kernel_missing is None or missing_kernel < best_kernel_missing):
                best_kernel_missing = missing_kernel
                best_kernel_state = RulerState(N, state.A)
        if state.missing_count > 50:
            targeted_prob = 0.9
        elif state.missing_count > 10:
            targeted_prob = 0.7
        else:
            targeted_prob = 0.4
        stage0 = step < stage0_steps
        if K and missing_kernel > 0:
            if stage0 and step % 200 == 0:
                ladder_proposals += 1
                moves, targets = forced_ladder_injection()
                if moves:
                    ladder_accepts += 1
                    frozen_targets = set(targets)
                    freeze_ttl = 2000
            if step % 500 == 0:
                comb_indices = [idx for idx in range(1, m - 1) if state.A[idx] % 8 == 0]
                if comb_indices:
                    idx = rng.choice(comb_indices)
                    for delta in (3, 5):
                        candidate = state.A[idx] + delta
                        if candidate >= N or candidate <= 0:
                            continue
                        if candidate in state.A:
                            continue
                        old_missing = state.missing_count
                        old_kernel = missing_kernel
                        old_residue = residue_score
                        old_pos, _new_pos, moved = state.move_mark(idx, candidate)
                        if moved:
                            new_kernel = missing_small_count(K)
                            new_residue = residue_penalty(8)
                            accept = False
                            if new_kernel < old_kernel:
                                accept = True
                            elif new_kernel == old_kernel and new_residue < old_residue:
                                accept = True
                            elif rng.random() < 0.02:
                                accept = True
                            if not accept:
                                state.move_mark(idx, old_pos)
                            temperature *= cooling
                            continue
            if rng.random() < 0.01:
                ladder_proposals += 1
                moves, old_missing, old_peak, _old_E, _old_spike, moved, targets = ladder_injection_move(K)
                if not moved:
                    continue
                new_kernel = missing_small_count(K)
                new_residue = residue_penalty(8)
                accept = False
                if new_kernel < missing_kernel:
                    accept = True
                elif new_kernel == missing_kernel and state.missing_count < old_missing:
                    accept = True
                elif new_kernel == missing_kernel and new_residue < residue_score:
                    accept = True
                elif new_kernel == missing_kernel and state.missing_count == old_missing and rng.random() < 0.02:
                    accept = True
                if not accept:
                    for idx, old_pos, _new_pos in sorted(moves, reverse=True):
                        state.move_mark(idx, old_pos)
                temperature *= cooling
                if accept:
                    ladder_accepts += 1
                    frozen_targets = set(targets)
                    freeze_ttl = 2000
                continue
            idx, old_pos, new_pos, old_missing, old_peak, _old_E, _old_spike, moved = (
                create_spacing_move(K)
            )
        else:
            idx, old_pos, new_pos, old_missing, old_peak, _old_E, _old_spike, moved = propose_move(
                targeted_prob
            )
        if not moved:
            continue
        old_h = ALPHA * old_missing
        new_h = ALPHA * state.missing_count
        accept = False
        new_kernel = missing_small_count(K) if K else 0
        if stage0 and K and missing_kernel > 0:
            if new_kernel < missing_kernel:
                accept = True
            elif rng.random() < 0.02:
                accept = True
        elif K and missing_kernel > 0:
            if new_kernel < missing_kernel:
                accept = True
            elif new_kernel == missing_kernel and state.missing_count < old_missing:
                accept = True
            elif new_kernel == missing_kernel and residue_penalty(8) < residue_score:
                accept = True
            elif new_kernel == missing_kernel and state.missing_count == old_missing and rng.random() < 0.05:
                accept = True
        elif state.missing_count < old_missing:
            accept = True
        elif state.missing_count == old_missing and rng.random() < 0.05:
            accept = True
        else:
            delta = new_h - old_h
            prob = math.exp(-delta / temperature)
            if rng.random() < prob:
                accept = True
        if accept:
            if state.missing_count == 0 and best_complete_state is None:
                best_complete_state = RulerState(N, state.A)
                best_complete_h = ALPHA * 0 + GAMMA * state.peak
        else:
            reverted_idx = state.A.index(new_pos)
            state.move_mark(reverted_idx, old_pos)
        temperature *= cooling

    # Phase 2: rigidity refinement (maintain completeness)
    if best_complete_state is not None:
        state = RulerState(N, best_complete_state.A)
        temperature = max(temperature, 1.0)
        for _ in range(phase2_steps):
            if rng.random() < 0.4:
                idx, old_pos, new_pos, old_missing, old_peak, old_E, old_spike, moved = (
                    try_spike_move()
                )
            else:
                idx, old_pos, new_pos, old_missing, old_peak, old_E, old_spike, moved = (
                    propose_move(0.4)
                )
            if not moved:
                continue
            if state.missing_count > 0:
                # revert immediately if completeness is broken
                reverted_idx = state.A.index(new_pos)
                state.move_mark(reverted_idx, old_pos)
                continue
            old_gap = old_E - energy_floor(state.S, N)
            new_gap = gap_from_state()
            new_spike = spike_mass_from_W(state.W)
            old_key = (old_gap, old_peak, old_spike)
            new_key = (new_gap, state.peak, new_spike)
            old_h = GAMMA * old_peak + BETA * old_E
            new_h = GAMMA * state.peak + BETA * state.E
            accept = False
            if new_key <= old_key:
                accept = True
            else:
                delta = new_h - old_h
                prob = math.exp(-delta / temperature)
                if rng.random() < prob:
                    accept = True
            if not accept:
                reverted_idx = state.A.index(new_pos)
                state.move_mark(reverted_idx, old_pos)
            else:
                best_state = RulerState(N, state.A)
                best_h = GAMMA * state.peak + BETA * state.E
            temperature *= cooling
        metrics = summarize_metrics(N, state.A, state.W)
        result = AnnealResult(
            A=state.A,
            metrics=metrics,
            H=best_h,
            kernel_A=best_kernel_state.A if best_kernel_state else None,
            kernel_M50=best_kernel_missing,
            ladder_proposals=ladder_proposals,
            ladder_accepts=ladder_accepts,
            best_kernel_seen=best_kernel_missing,
        )
        if os.environ.get("SPARSE_RULER_DEBUG"):
            print(
                "DEBUG anneal_run",
                "best_kernel",
                best_kernel_missing,
                "ladder_proposals",
                ladder_proposals,
                "ladder_accepts",
                ladder_accepts,
            )
        return result

    metrics = summarize_metrics(N, best_state.A, best_state.W)
    result = AnnealResult(
        A=best_state.A,
        metrics=metrics,
        H=best_h,
        kernel_A=best_kernel_state.A if best_kernel_state else None,
        kernel_M50=best_kernel_missing,
        ladder_proposals=ladder_proposals,
        ladder_accepts=ladder_accepts,
        best_kernel_seen=best_kernel_missing,
    )
    if os.environ.get("SPARSE_RULER_DEBUG"):
        print(
            "DEBUG anneal_run",
            "best_kernel",
            best_kernel_missing,
            "ladder_proposals",
            ladder_proposals,
            "ladder_accepts",
            ladder_accepts,
        )
    return result


def refine_solution(
    N: int,
    A: Sequence[int],
    steps: int,
    rng: random.Random,
    temperature: float = 5000.0,
    cooling: float = 0.9995,
) -> AnnealResult:
    state = RulerState(N, A)
    if state.missing_count != 0:
        raise ValueError("refine_solution requires a complete ruler")

    def spike_mass_from_W(W: Sequence[int]) -> int:
        return sum(max(w - 2, 0) for w in W[1:])

    def try_spike_move() -> Tuple[int, int, int, int, int, int, int, bool]:
        defect_distances = [d for d in range(1, N + 1) if state.W[d] >= 4]
        if not defect_distances:
            defect_distances = [d for d in range(1, N + 1) if state.W[d] >= 3]
        if not defect_distances:
            return 0, 0, 0, 0, 0, 0, 0, False
        target_d = rng.choice(defect_distances)
        indices = list(range(1, state.m - 1))
        rng.shuffle(indices)
        occupied = set(state.A)
        for i in indices:
            ai = state.A[i]
            for j in range(state.m):
                if j == i:
                    continue
                aj = state.A[j]
                if abs(ai - aj) != target_d:
                    continue
                deltas = [-3, -2, -1, 1, 2, 3]
                rng.shuffle(deltas)
                for delta in deltas:
                    candidate = ai + delta
                    if not (1 <= candidate <= N - 1):
                        continue
                    if candidate in occupied:
                        continue
                    old_missing = state.missing_count
                    old_peak = state.peak
                    old_E = state.E
                    old_spike = spike_mass_from_W(state.W)
                    old_pos, new_pos, moved = state.move_mark(i, candidate)
                    return i, old_pos, new_pos, old_missing, old_peak, old_E, old_spike, moved
        return 0, 0, 0, 0, 0, 0, 0, False

    best_state = RulerState(N, state.A)
    best_key = (
        state.E - energy_floor(state.S, N),
        state.peak,
        spike_mass_from_W(state.W),
    )
    for _ in range(steps):
        idx, old_pos, new_pos, old_missing, old_peak, old_E, old_spike, moved = (
            try_spike_move()
        )
        if not moved:
            continue
        if state.missing_count > 0:
            reverted_idx = state.A.index(new_pos)
            state.move_mark(reverted_idx, old_pos)
            continue
        old_key = (
            old_E - energy_floor(state.S, N),
            old_peak,
            old_spike,
        )
        new_key = (
            state.E - energy_floor(state.S, N),
            state.peak,
            spike_mass_from_W(state.W),
        )
        accept = False
        if new_key <= old_key:
            accept = True
        else:
            delta = (state.E - old_E) + (state.peak - old_peak) * GAMMA
            prob = math.exp(-delta / temperature)
            if rng.random() < prob:
                accept = True
        if accept:
            best_state = RulerState(N, state.A)
            best_key = new_key
        else:
            reverted_idx = state.A.index(new_pos)
            state.move_mark(reverted_idx, old_pos)
        temperature *= cooling

    metrics = summarize_metrics(N, best_state.A, best_state.W)
    H = GAMMA * metrics.peak + BETA * metrics.E
    return AnnealResult(A=best_state.A, metrics=metrics, H=H)


def _run_single(args: Tuple[int, int, int, int, float, float, int]) -> AnnealResult:
    N, m, steps, seed, temperature, cooling, stop_rho = args
    rng = random.Random(seed)
    return anneal_run(
        N=N,
        m=m,
        steps=steps,
        rng=rng,
        temperature=temperature,
        cooling=cooling,
        stop_rho=stop_rho,
    )


def run_experiments(
    N: int,
    m: int,
    runs: int,
    steps: int,
    seed: Optional[int] = None,
    processes: int = 1,
    temperature: float = 5000.0,
    cooling: float = 0.9995,
    stop_rho: float = 1.05,
) -> Tuple[List[AnnealResult], Optional[ImpossibleResult]]:
    S = m * (m - 1) // 2
    if S < N:
        return [], ImpossibleResult(N=N, m=m, S=S)
    rng = random.Random(seed)
    seeds = [rng.randint(0, 2**31 - 1) for _ in range(runs)]
    args = [(N, m, steps, seeds[i], temperature, cooling, stop_rho) for i in range(runs)]
    if processes > 1:
        import multiprocessing as mp

        with mp.Pool(processes=processes) as pool:
            results = pool.map(_run_single, args)
    else:
        results = [_run_single(arg) for arg in args]
    return results, None


def best_solutions(results: Sequence[AnnealResult], k: int = 5) -> List[AnnealResult]:
    return sorted(results, key=lambda r: r.H)[:k]


def write_best_solutions(
    results: Sequence[AnnealResult],
    out_dir: Path,
    N: int,
    m: int,
    k: int = 5,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    best = best_solutions(results, k=k)
    paths = []
    for idx, result in enumerate(best, start=1):
        payload = {
            "N": N,
            "m": m,
            "A": result.A,
            "metrics": result.metrics.__dict__,
        }
        path = out_dir / f"solution_N{N}_m{m}_{idx}.json"
        path.write_text(json.dumps(payload, indent=2))
        paths.append(path)
    return paths


def write_csv(
    rows: Iterable[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = format_csv_header()
    path.write_text("\n".join([header, *rows]) + "\n")


def summarize_best(result: AnnealResult) -> MetricSummary:
    return result.metrics


def scan_range(
    Ns: Iterable[int],
    m_values: Iterable[int],
    runs: int,
    steps: int,
    out_dir: Path,
    seed: Optional[int] = None,
    processes: int = 1,
) -> Path:
    rows = []
    for N in Ns:
        for m in m_values:
            results, impossible = run_experiments(
                N=N,
                m=m,
                runs=runs,
                steps=steps,
                seed=seed,
                processes=processes,
            )
            if impossible:
                rows.append(impossible.to_csv_row())
                continue
            best = best_solutions(results, k=1)[0]
            rows.append(best.metrics.to_csv_row())
            write_best_solutions(results, out_dir / "solutions", N, m)
    csv_path = out_dir / "results.csv"
    write_csv(rows, csv_path)
    return csv_path


def default_small_scan(out_dir: Path, runs: int, steps: int, seed: Optional[int], processes: int) -> Path:
    Ns = list(range(1, 121))
    rows = []
    for N in Ns:
        m0 = math.ceil(math.sqrt(3 * N))
        for m in (m0, m0 + 1, m0 + 2):
            results, impossible = run_experiments(
                N=N,
                m=m,
                runs=runs,
                steps=steps,
                seed=seed,
                processes=processes,
            )
            if impossible:
                rows.append(impossible.to_csv_row())
                continue
            best = best_solutions(results, k=1)[0]
            rows.append(best.metrics.to_csv_row())
            write_best_solutions(results, out_dir / "solutions", N, m)
    csv_path = out_dir / "small_scan.csv"
    write_csv(rows, csv_path)
    return csv_path


def default_large_scan(out_dir: Path, runs: int, steps: int, seed: Optional[int], processes: int) -> Path:
    Ns = [1000, 2000, 5000, 10000, 20000, 50000]
    rows = []
    for N in Ns:
        m0 = math.ceil(math.sqrt(3 * N))
        for m in (m0, m0 + 1, m0 + 2):
            results, impossible = run_experiments(
                N=N,
                m=m,
                runs=runs,
                steps=steps,
                seed=seed,
                processes=processes,
            )
            if impossible:
                rows.append(impossible.to_csv_row())
                continue
            best = best_solutions(results, k=1)[0]
            rows.append(best.metrics.to_csv_row())
            write_best_solutions(results, out_dir / "solutions", N, m)
    csv_path = out_dir / "large_scan.csv"
    write_csv(rows, csv_path)
    return csv_path
