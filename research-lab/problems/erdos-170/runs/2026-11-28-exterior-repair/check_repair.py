"""Exact integer-set sieve for deleting two marks and inserting one fresh mark."""

import argparse
from itertools import combinations


def blocks(r):
    a = 2 * r + 1
    b = a + 1
    q = a + b
    h = (3 * r + 1) * q
    return (
        list(range(r + 1)),
        [i * a for i in range(1, r + 1)],
        [(r + 1) * a + j * q for j in range(2 * r + 1)],
        [h - i * b for i in range(1, r + 1)],
        list(range(h, h + r + 1)),
    )


def bits(marks):
    return sum(1 << x for x in marks)


def repair_mask(original, u, v):
    """The set bit z means precisely that (W - {u,v}) + {z} is complete."""
    n = max(original)
    surviving = original - {u, v}
    marks = bits(surviving)
    domain = (1 << (n + 1)) - 1
    candidates = domain ^ bits(original)
    distances = 0
    for x in surviving:
        distances |= marks >> x
    missing = domain & ~distances
    while candidates and missing:
        least = missing & -missing
        d = least.bit_length() - 1
        candidates &= (marks << d) | (marks >> d)
        missing ^= least
    return candidates


def direct_mask(original, u, v):
    n = max(original)
    target = set(range(1, n + 1))
    valid = []
    for z in set(range(n + 1)) - original:
        repaired = (original - {u, v}) | {z}
        if {abs(x - y) for x, y in combinations(repaired, 2)} == target:
            valid.append(z)
    return bits(valid)


def controls():
    toy = {0, 1, 2, 3, 5}
    assert repair_mask(toy, 1, 2) == direct_mask(toy, 1, 2) == bits([4])
    for n in range(3, 8):
        for mask in range(1 << (n - 1)):
            w = {0, n} | {x for x in range(1, n) if mask >> (x - 1) & 1}
            for u, v in combinations(w, 2):
                if len(w) > 2:
                    assert repair_mask(w, u, v) == direct_mask(w, u, v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-r', type=int, default=25)
    args = parser.parse_args()
    controls()
    print('r  exterior_pairs  exterior_repairs  all_pairs  all_repairs')
    for r in range(1, args.max_r + 1):
        b0, b1, b2, b3, b4 = blocks(r)
        e = set(b0 + b1 + b3 + b4)
        w = e | set(b2)
        assert len(w) == 6 * r + 3 and max(w) == 12 * r * r + 14 * r + 3
        ne = na = exterior_repairs = all_repairs = 0
        for u, v in combinations(sorted(w), 2):
            repairs = repair_mask(w, u, v).bit_count()
            na += 1
            all_repairs += repairs
            if u in e and v in e:
                ne += 1
                exterior_repairs += repairs
        print(r, ne, exterior_repairs, na, all_repairs, flush=True)
        assert all_repairs == 0


if __name__ == '__main__':
    main()
