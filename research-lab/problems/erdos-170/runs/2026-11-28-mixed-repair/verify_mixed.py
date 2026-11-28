#!/usr/bin/env python3
"""Literal verifier and bounded mixed-repair search for canonical W(r,2r)."""

from argparse import ArgumentParser
from collections import Counter
from itertools import accumulate, combinations


def ruler(r):
    if r < 1:
        raise ValueError("r must be a positive integer")
    a, b = 2 * r + 1, 2 * r + 2
    q = a + b
    length = (3 * r + 1) * q + r
    left_junction = (r + 1) * a
    blocks = (
        tuple(range(r + 1)),
        tuple(i * a for i in range(1, r + 1)),
        tuple(left_junction + j * q for j in range(2 * r + 1)),
        tuple(length - r - i * b for i in range(r, 0, -1)),
        tuple(range(length - r, length + 1)),
    )
    marks = tuple(mark for block in blocks for mark in block)
    gaps = [1] * r + [r + 1] + [a] * r + [q] * (2 * r) + [b] * (r + 1) + [1] * r
    assert marks == tuple(accumulate(gaps, initial=0))
    assert len(marks) == 6 * r + 3
    return length, blocks, marks


def missing_differences(marks, length):
    """Check the literal interval and compute uncovered lengths from all pairs."""
    if any(type(mark) is not int or not 0 <= mark <= length for mark in marks):
        raise ValueError("every mark must be an integer in the original interval")
    if len(marks) != len(set(marks)):
        raise ValueError("marks must be distinct")
    realized = {right - left for left, right in combinations(sorted(marks), 2)}
    return set(range(1, length + 1)) - realized


def check_mixed_witness(r, middle, exterior, inserted):
    length, blocks, marks = ruler(r)
    if middle not in blocks[2][1:-1]:
        raise ValueError("the first deletion must be a strict middle mark")
    if exterior not in blocks[0] + blocks[1] + blocks[3] + blocks[4]:
        raise ValueError("the second deletion must be exterior")
    if inserted in marks:
        raise ValueError("the inserted mark must be fresh relative to the original ruler")
    repaired = tuple(mark for mark in marks if mark not in (middle, exterior)) + (inserted,)
    return missing_differences(repaired, length)


def controls():
    length, _, marks = ruler(1)
    assert missing_differences(marks, length) == set()
    assert length in missing_differences(marks[1:], length)
    # Distances 1 through 6 are all covered despite gaps between marks.
    assert missing_differences((0, 1, 4, 6), 6) == set()
    for broken in ((0, 1, 1, 6), (0, 1, 4, 7)):
        try:
            missing_differences(broken, 6)
        except ValueError:
            pass
        else:
            raise AssertionError(f"checker accepted invalid marks {broken}")
    print("PASS controls: complete rulers, deleted endpoint, repeated and out-of-range marks")

    length, blocks, marks = ruler(2)
    strict_middle = blocks[2][1:-1]
    original = set(marks)
    fresh = tuple(mark for mark in range(length + 1) if mark not in original)
    for deleted_count, inserted_count in ((2, 1), (3, 2)):
        checked = 0
        for deleted in combinations(strict_middle, deleted_count):
            kept = tuple(mark for mark in marks if mark not in deleted)
            for inserted in combinations(fresh, inserted_count):
                assert missing_differences(kept + inserted, length)
                checked += 1
        total_description = f"delete {deleted_count} strict middle, insert {inserted_count} fresh"
        print(f"PASS central regression r=2: {total_description}, {checked} cases")

    kept = tuple(mark for mark in marks if mark not in (26, 2))
    lost = missing_differences(kept, length)
    assert lost == {3, 16, 21, 24, 25, 26, 35, 39, 45, 46, 51, 52, 53, 57, 75}
    print("PASS loss control: (r,middle,exterior)=(2,26,2) loses exactly 15 lengths")


def possible_insertions(length, distance, kept, original):
    return {
        inserted
        for mark in kept
        for inserted in (mark - distance, mark + distance)
        if 0 <= inserted <= length and inserted not in original
    }


def scan(r, totals, minimum_by_block):
    length, blocks, marks = ruler(r)
    assert not missing_differences(marks, length)
    pair_counts = Counter(right - left for left, right in combinations(marks, 2))
    incident = {
        mark: Counter(abs(other - mark) for other in marks if other != mark)
        for mark in marks
    }
    original = frozenset(marks)
    for middle in blocks[2][1:-1]:
        for block in (0, 1, 3, 4):
            for exterior in blocks[block]:
                kept = original - {middle, exterior}
                removed = incident[middle] + incident[exterior]
                removed[abs(middle - exterior)] -= 1  # counted at both ends
                lost = {distance for distance in removed if removed[distance] == pair_counts[distance]}
                assert lost
                if r <= 4:
                    assert lost == missing_differences(tuple(kept), length)
                totals["deletion_pairs"] += 1
                totals["all_eligible_insertions"] += length + 1 - len(original)
                if exterior not in (0, length):
                    result = (len(lost) - 6 * r, r, middle, exterior)
                    minimum_by_block[block] = min(minimum_by_block.get(block, result), result)
                # Each formerly covered length missing in the kept marks must be
                # supplied by an edge from the inserted mark to a kept mark.
                candidates = possible_insertions(length, max(lost), kept, original)
                totals["max_lost_length_candidates"] += len(candidates)
                for inserted in sorted(candidates):
                    covered_by_new = {abs(inserted - mark) for mark in kept}
                    if lost <= covered_by_new:
                        totals["literal_witness_checks"] += 1
                        if not check_mixed_witness(r, middle, exterior, inserted):
                            return r, middle, exterior, inserted
                        raise AssertionError("necessary and sufficient sieve disagrees with literal checker")
    return None


def main():
    parser = ArgumentParser(description=__doc__)
    queries = parser.add_mutually_exclusive_group()
    queries.add_argument("--max-r", type=int, help="exhaustively test each 1 <= r <= this bound")
    queries.add_argument("--witness", type=int, nargs=4, metavar=("R", "MIDDLE", "EXTERIOR", "INSERTED"))
    args = parser.parse_args()
    controls()
    if args.witness:
        missing = check_mixed_witness(*args.witness)
        print(f"witness={tuple(args.witness)} missing_count={len(missing)} first_missing={min(missing, default=None)}")
        raise SystemExit(bool(missing))
    max_r = 12 if args.max_r is None else args.max_r
    if max_r < 1:
        parser.error("--max-r must be positive")
    totals = Counter(deletion_pairs=0, all_eligible_insertions=0, max_lost_length_candidates=0, literal_witness_checks=0)
    minimum_by_block = {}
    for r in range(1, max_r + 1):
        witness = scan(r, totals, minimum_by_block)
        if witness:
            print(f"COUNTEREXAMPLE: {witness}")
            return
    print(f"PASS finite exclusion: 1 <= r <= {max_r}")
    for name, total in totals.items():
        print(f"{name}={total}")
    print("minimum nonendpoint losses minus 6r by block (gap, r, middle, exterior):")
    for block, result in sorted(minimum_by_block.items()):
        print(f"B{block}: {result}")


if __name__ == "__main__":
    main()
