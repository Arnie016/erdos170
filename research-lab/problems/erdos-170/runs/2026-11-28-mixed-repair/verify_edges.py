"""Check the proof's designated unique-length edges from literal ruler gaps."""

from collections import Counter
from itertools import accumulate, combinations


def check(r):
    a, b = 2 * r + 1, 2 * r + 2
    gaps = [1] * r + [r + 1] + [a] * r + [a + b] * (2 * r) + [b] * (r + 1) + [1] * r
    marks = tuple(accumulate(gaps, initial=0))
    length = 12 * r * r + 14 * r + 3
    assert len(marks) == len(set(marks)) == 6 * r + 3
    assert marks[0] == 0 and marks[-1] == length
    assert all(0 <= x <= length for x in marks)

    left_cap = marks[: r + 1]
    left_progression = marks[r + 1 : 2 * r + 1]
    middle = marks[2 * r + 1 : 4 * r + 2]
    right_progression = marks[4 * r + 2 : 5 * r + 2]
    right_cap = marks[5 * r + 2 :]
    exterior = left_cap + left_progression + right_progression + right_cap
    strict = middle[1:-1]
    left_junction, right_junction = middle[0], middle[-1]

    counts = Counter(y - x for x, y in combinations(marks, 2))
    assert counts.keys() == set(range(1, length + 1))
    for u in strict:
        for v in exterior:
            assert counts[abs(u - v)] == 1, (r, u, v)

    witnesses = (
        {v: (a, right_junction) for v in left_cap[1:]}
        | {v: (length - r - b, right_junction) for v in left_progression}
        | {v: (a, left_junction) for v in right_progression}
        | {v: (a, right_junction) for v in right_cap[:-1]}
    )
    assert set(witnesses) == set(exterior) - {0, length}
    for v, partners in witnesses.items():
        assert len({v, *partners}) == 3
        for x in partners:
            assert x in marks and x not in strict
            assert counts[abs(x - v)] == 1, (r, x, v)
    return len(strict) * len(exterior), 2 * len(witnesses)


if __name__ == "__main__":
    parameters = (*range(1, 33), 65)
    base_counts, extra_counts = zip(*(check(r) for r in parameters))
    print(
        f"PASS r=1..32 and r=65: {sum(base_counts):,} strict-middle/exterior "
        f"edges and {sum(extra_counts):,} designated additional edges have unique lengths per check"
    )
