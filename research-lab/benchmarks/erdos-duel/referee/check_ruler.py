#!/usr/bin/env python3
"""Check each eligible finite repair with direct integer bitsets."""

import argparse
from itertools import accumulate, combinations


def ruler(r):
    a, b = 2 * r + 1, 2 * r + 2
    gaps = [1] * r + [r + 1] + [a] * r + [a + b] * (2 * r) + [b] * (r + 1) + [1] * r
    marks = tuple(accumulate(gaps, initial=0))
    if len(marks) != 6 * r + 3 or len(marks) != len(set(marks)):
        raise RuntimeError("Invalid canonical marks")
    first_middle = 2 * r + 1
    past_middle = first_middle + 2 * r + 1
    return marks, marks[:first_middle] + marks[past_middle:]


def mask_of(marks):
    return sum(1 << mark for mark in marks)


def differences(marks, mask):
    result = 0
    for mark in marks:
        result |= mask >> mark
    return result


def incident(inserted, kept, reflected, length):
    return (kept >> inserted) | (reflected >> (length - inserted))


def required_mask(length):
    return (1 << (length + 1)) - 2


def controls():
    kept = (0, 1, 6)
    length = 6
    bits = mask_of(kept)
    reflected = mask_of(length - x for x in kept)
    lost = required_mask(length) & ~differences(kept, bits)
    if lost & ~incident(4, bits, reflected, length):
        raise RuntimeError("Known full repair failed")
    if not lost & ~incident(3, bits, reflected, length):
        raise RuntimeError("Deliberately broken repair passed")
    marks, _ = ruler(1)
    if not required_mask(marks[-1]) & ~differences(marks[1:], mask_of(marks[1:])):
        raise RuntimeError("Deliberately missing endpoint passed")
    print("PASS full repair, broken repair, and missing endpoint controls")


def examine(r, any_pair):
    marks, exterior = ruler(r)
    length = marks[-1]
    if length != (3 * r + 1) * (4 * r + 3) + r:
        raise RuntimeError("Literal length mismatch")
    required = required_mask(length)
    original = mask_of(marks)
    if required & ~differences(marks, original):
        raise RuntimeError("Original ruler is incomplete")
    fresh = tuple(z for z in range(length + 1) if not original & (1 << z))
    checked_pairs = 0
    for u, v in combinations(marks if any_pair else exterior, 2):
        kept = tuple(mark for mark in marks if mark != u and mark != v)
        bits = original ^ ((1 << u) | (1 << v))
        reflected = mask_of(length - x for x in kept)
        lost = required & ~differences(kept, bits)
        for z in fresh:
            if not lost & ~incident(z, bits, reflected, length):
                all_marks = sorted((*kept, z))
                literal = {b - a for a, b in combinations(all_marks, 2)}
                if literal != set(range(1, length + 1)):
                    raise RuntimeError("Bitset and pair check disagree")
                raise RuntimeError(f"Counterexample r={r}, delete=({u},{v}), insert={z}")
        checked_pairs += 1
    return checked_pairs, checked_pairs * len(fresh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-r", type=int, default=9)
    parser.add_argument("--any-pair", action="store_true", help="include strict middle and junction deletions")
    args = parser.parse_args()
    if args.max_r < 1:
        parser.error("--max-r must be positive")
    controls()
    total_pairs = 0
    total_insertions = 0
    for r in range(1, args.max_r + 1):
        pairs, insertions = examine(r, args.any_pair)
        print(f"r={r} pairs={pairs} fresh insertion trials={insertions}")
        total_pairs += pairs
        total_insertions += insertions
    scope = "all distinct original-mark pairs" if args.any_pair else "exterior pairs"
    print(f"PASS {scope}, 1<=r<={args.max_r}, pairs={total_pairs}, insertions={total_insertions}")


if __name__ == "__main__":
    main()
