from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from sparse_ruler.metrics import compute_W
from tools.patch_missing import PatchCandidate, score_candidates


@dataclass
class PatchStep:
    step: int
    moved_from: int
    moved_to: int
    old_missing_count: int
    new_missing_count: int
    old_missing: List[int]
    new_missing: List[int]


def parse_int_list(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def missing_list(N: int, marks: Sequence[int]) -> List[int]:
    W = compute_W(marks, N)
    return [d for d in range(1, N + 1) if W[d] == 0]


def weighted_missing_sum(missing: Sequence[int]) -> int:
    return sum(d * d for d in missing)


def anchors_from_marks(marks: Sequence[int]) -> List[int]:
    anchors = [m for m in marks if m <= 80]
    extra = [m for m in marks if m in {181, 196, 197, 199, 200}]
    if extra:
        anchors.extend(extra)
    else:
        anchors.extend(sorted(marks)[-5:])
    return list(dict.fromkeys(anchors))


def responsibility_scores(N: int, marks: Sequence[int]) -> dict[int, int]:
    W = compute_W(marks, N)
    scores = {mark: 0 for mark in marks}
    for i, ai in enumerate(marks):
        for j in range(i + 1, len(marks)):
            aj = marks[j]
            d = abs(aj - ai)
            if W[d] == 1:
                scores[ai] += 1
                scores[aj] += 1
    return scores


def select_move_from_marks(
    N: int,
    marks: Sequence[int],
    scaffold_threshold: int,
    candidate_position: int,
) -> int:
    scaffold = [m for m in marks if m > scaffold_threshold and m not in {0, N}]
    scores = responsibility_scores(N, marks)
    scaffold.sort(key=lambda m: scores.get(m, 0))
    for mark in scaffold:
        if mark != candidate_position:
            return mark
    raise ValueError("No movable scaffold mark available")


def apply_move(marks: Sequence[int], old_mark: int, new_mark: int) -> List[int]:
    updated = [m for m in marks if m != old_mark]
    updated.append(new_mark)
    return sorted(updated)


def choose_candidate(candidates: Iterable[PatchCandidate]) -> PatchCandidate | None:
    candidates = list(candidates)
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c.gain, sum(c.fixes)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy patch loop using patch_missing scoring.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--marks", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--scaffold-threshold", type=int, default=80)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--log", type=str, default="patch_trace.jsonl")
    args = parser.parse_args()

    N = args.n
    marks = parse_int_list(args.marks)
    log_entries: List[PatchStep] = []

    for step in range(1, args.max_steps + 1):
        old_missing = missing_list(N, marks)
        old_missing_count = len(old_missing)
        if old_missing_count == 0:
            break
        anchors = anchors_from_marks(marks)
        candidates = score_candidates(N, marks, old_missing, anchors)[: args.top]
        candidate = choose_candidate(candidates)
        if candidate is None:
            break
        move_from = select_move_from_marks(N, marks, args.scaffold_threshold, candidate.position)
        proposed = apply_move(marks, move_from, candidate.position)
        new_missing = missing_list(N, proposed)
        new_missing_count = len(new_missing)
        accept = False
        if new_missing_count < old_missing_count:
            accept = True
        else:
            old_weighted = weighted_missing_sum(old_missing)
            new_weighted = weighted_missing_sum(new_missing)
            if new_weighted < old_weighted:
                accept = True
            elif new_missing_count == old_missing_count:
                removed = set(old_missing) - set(new_missing)
                if any(d >= 170 for d in removed):
                    accept = True
        if accept:
            marks = proposed
            log_entries.append(
                PatchStep(
                    step=step,
                    moved_from=move_from,
                    moved_to=candidate.position,
                    old_missing_count=old_missing_count,
                    new_missing_count=new_missing_count,
                    old_missing=old_missing,
                    new_missing=new_missing,
                )
            )
        else:
            break

    with open(args.log, "w", encoding="utf-8") as handle:
        for entry in log_entries:
            handle.write(json.dumps(entry.__dict__))
            handle.write("\n")

    print(json.dumps({"final_marks": marks, "missing": missing_list(N, marks)}, indent=2))


if __name__ == "__main__":
    main()
