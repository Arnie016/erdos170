#!/usr/bin/env python3
"""Find and verify literal clique and cover certificates over F3."""

from itertools import combinations, product


FIELD = (0, 1, 2)
ZERO = (0, 0, 0, 0)
GRAPH = ((0, 1), (1, 2))  # vertex 3 is isolated


def scale(scalar, vector):
    return tuple(scalar * entry % 3 for entry in vector)


def add(x, y):
    return tuple((a + b) % 3 for a, b in zip(x, y))


def orthogonal(x, y):
    return all((x[i] * y[j] - x[j] * y[i]) % 3 == 0 for i, j in GRAPH)


def span(vectors):
    space = {ZERO}
    for generator in vectors:
        space = {add(x, scale(scalar, generator)) for x in space for scalar in FIELD}
    return frozenset(space)


def check_certificates(clique, bases):
    universe = set(product(FIELD, repeat=len(ZERO)))
    if len(clique) != len(set(clique)) or any(x not in universe for x in clique):
        return False
    if any(orthogonal(x, y) for x, y in combinations(clique, 2)):
        return False
    spaces = [span(basis) for basis in bases]
    if any(not orthogonal(x, y) for space in spaces for x, y in combinations(sorted(space), 2)):
        return False
    return set().union(*spaces) == universe and len(clique) == len(bases)


def canonical(vector):
    first = next(entry for entry in vector if entry)
    return scale(first, vector)  # each nonzero scalar in F3 is its own inverse


def maximal_isotropic_spaces(projective):
    spaces = {frozenset({ZERO}): ()}
    pending = [frozenset({ZERO})]
    maximal = {}
    while pending:
        space = pending.pop()
        basis = spaces[space]
        extensions = [x for x in projective if x not in space and all(orthogonal(x, y) for y in basis)]
        if not extensions:
            maximal[space] = basis
        for x in extensions:
            enlarged = span((*basis, x))
            if enlarged not in spaces:
                spaces[enlarged] = (*basis, x)
                pending.append(enlarged)
    return maximal, len(spaces)


def maximum_clique(vectors):
    adjacency = []
    for x in vectors:
        adjacency.append(sum(1 << j for j, y in enumerate(vectors) if not orthogonal(x, y)))
    best = ()

    def extend(clique, remaining):
        nonlocal best
        if len(clique) > len(best):
            best = clique
        while len(clique) + remaining.bit_count() > len(best):
            vertex_bit = remaining & -remaining
            vertex = vertex_bit.bit_length() - 1
            remaining ^= vertex_bit
            extend((*clique, vertex), remaining & adjacency[vertex])

    extend((), (1 << len(vectors)) - 1)
    return tuple(vectors[i] for i in best)


def equal_cover_size(vectors, maximal, size):
    masks = [
        sum(1 << i for i, x in enumerate(vectors) if x in space)
        for space in maximal
    ]
    bases = tuple(maximal.values())
    membership = [[i for i, mask in enumerate(masks) if mask & (1 << j)] for j in range(len(vectors))]

    def cover(remaining, chosen):
        if not remaining:
            return chosen
        if len(chosen) == size:
            return None
        first = min((j for j in range(len(vectors)) if remaining & (1 << j)), key=lambda j: len(membership[j]))
        for i in sorted(membership[first], key=lambda i: -(masks[i] & remaining).bit_count()):
            result = cover(remaining & ~masks[i], (*chosen, bases[i]))
            if result is not None:
                return result
        return None

    return cover((1 << len(vectors)) - 1, ())


def main():
    vectors = tuple(sorted({canonical(v) for v in product(FIELD, repeat=4) if v != ZERO}))
    spaces, space_count = maximal_isotropic_spaces(vectors)
    clique = maximum_clique(vectors)
    bases = equal_cover_size(vectors, spaces, len(clique))
    if bases is None or not check_certificates(clique, bases):
        raise RuntimeError("No valid matching clique and cover found")
    if check_certificates(clique[:-1], bases[:-1]):
        raise RuntimeError("Broken cover passed")
    if check_certificates((*clique[:-1], scale(2, clique[0])), bases):
        raise RuntimeError("Broken clique passed")
    print(f"F3; edges={GRAPH}; isolated vertex=3")
    print(f"Projective points={len(vectors)}; isotropic spaces={space_count}; maximal spaces={len(spaces)}")
    print(f"Exact clique and cover size={len(clique)}")
    print(f"Clique={clique}")
    print(f"Cover bases={bases}")
    print("PASS literal certificate and deliberately broken clique and cover controls")
    if len(clique) != 10:
        raise RuntimeError("Exact check disagrees with weighted support prediction 10")


if __name__ == "__main__":
    main()
