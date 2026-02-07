from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class PatchCandidate:
    position: int
    fixes: List[int]

    @property
    def gain(self) -> int:
        return len(self.fixes)


def parse_int_list(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def candidate_fixes(position: int, anchors: Iterable[int], missing_set: set[int]) -> List[int]:
    fixes = []
    for anchor in anchors:
        d = abs(position - anchor)
        if d in missing_set:
            fixes.append(d)
    return sorted(set(fixes))


def score_candidates(
    N: int,
    marks: Sequence[int],
    missing: Sequence[int],
    anchors: Sequence[int],
) -> List[PatchCandidate]:
    missing_set = set(missing)
    occupied = set(marks)
    candidates = []
    for position in range(1, N):
        if position in occupied:
            continue
        fixes = candidate_fixes(position, anchors, missing_set)
        if not fixes:
            continue
        candidates.append(PatchCandidate(position=position, fixes=fixes))
    candidates.sort(key=lambda c: (c.gain, sum(c.fixes)), reverse=True)
    return candidates


def format_candidate(candidate: PatchCandidate) -> str:
    fixes = ",".join(str(d) for d in candidate.fixes)
    return f"{candidate.position}\t{candidate.gain}\t[{fixes}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidate patch positions for missing distances.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--marks", type=str, required=True)
    parser.add_argument("--missing", type=str, required=True)
    parser.add_argument("--anchors", type=str, required=True)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    marks = parse_int_list(args.marks)
    missing = parse_int_list(args.missing)
    anchors = parse_int_list(args.anchors)
    candidates = score_candidates(args.n, marks, missing, anchors)

    print("position\tgain\tfixes")
    for candidate in candidates[: args.top]:
        print(format_candidate(candidate))


if __name__ == "__main__":
    main()
