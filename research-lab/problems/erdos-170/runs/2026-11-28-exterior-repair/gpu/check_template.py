"""Check the exact deletion and insertion complements using only literal mark pairs."""
from itertools import combinations


def marks_from_gaps(r):
    gaps = [1] * r + [r + 1] + [2 * r + 1] * r + [4 * r + 3] * (2 * r)
    gaps += [2 * r + 2] * (r + 1) + [1] * r
    marks = [0]
    for step in gaps:
        marks.append(marks[-1] + step)
    return marks


def missing(marks, length):
    present = {abs(x - y) for x, y in combinations(marks, 2)}
    return set(range(1, length + 1)) - present


def check(r):
    marks = marks_from_gaps(r)
    a = 2 * r + 1
    b = a + 1
    q = a + b
    t = 2 * r * q
    left = (r + 1) * a
    right = left + t
    f = (r + 1) * b
    z = left + b
    kept = [x for x in marks if x not in (left, right)]
    if len(kept) != len(marks) - 2 or z in marks:
        raise ValueError(f"Invalid edit at r={r}")
    residual = {left, f, t - q, right, t + f} | {t + k * a for k in range(1, r + 1)}
    deletion = residual | {t + k * b for k in range(r + 1)}
    deletion |= {right - x for x in range(1, r + 1)} | {f + x for x in range(1, r + 1)}
    before = missing(kept, marks[-1])
    after = missing(kept + [z], marks[-1])
    if r == 1:
        if before != deletion or after != residual - {t - q} or len(after) != 5:
            raise ValueError("The r=1 exception for this specific edit changed")
    elif before != deletion or len(deletion) != 4 * r + 6 or after != residual or len(residual) != r + 5:
        raise ValueError(f"Wrong missing-distance set at r={r}")


if __name__ == "__main__":
    for radius in range(1, 101):
        check(radius)
    print("PASS exact pre-insertion and post-insertion complements through r=100, including r=1 exception")
