"""Literal determinant check for two disjoint edges and an isolated vertex."""

from itertools import combinations, product


FIELD = range(3)
VERTICES = frozenset(range(5))
EDGES = ((0, 1), (2, 3))
VECTOR_SPACE = frozenset(product(FIELD, repeat=len(VERTICES)))


def orthogonal(left, right):
    return all(
        (left[i] * right[j] - left[j] * right[i]) % 3 == 0 for i, j in EDGES
    )


def span(basis):
    return {
        tuple(
            sum(coefficient * row[i] for coefficient, row in zip(scalars, basis)) % 3
            for i in range(len(VERTICES))
        )
        for scalars in product(FIELD, repeat=len(basis))
    }


def check_witness(clique, bases):
    if len(set(clique)) != len(clique) or not set(clique) <= VECTOR_SPACE:
        raise ValueError("invalid clique points")
    if any(orthogonal(left, right) for left, right in combinations(clique, 2)):
        raise ValueError("orthogonal pair in clique")
    covered = set()
    for basis in bases:
        if not set(basis) <= VECTOR_SPACE:
            raise ValueError("invalid basis points")
        space = span(basis)
        if len(space) != 3 ** len(basis):
            raise ValueError("dependent basis")
        if any(not orthogonal(left, right) for left, right in combinations(space, 2)):
            raise ValueError("subspace is not isotropic")
        covered.update(space)
    if covered != VECTOR_SPACE:
        raise ValueError("cover misses field vectors")
    if len(clique) != len(bases):
        raise ValueError("clique and cover sizes differ")


def weighted_support_count():
    neighbors = {v: set() for v in VERTICES}
    for i, j in EDGES:
        neighbors[i].add(j)
        neighbors[j].add(i)

    total = 0
    supports = 0
    for size in range(len(VERTICES) + 1):
        for marks in combinations(sorted(VERTICES), size):
            support = frozenset(marks)
            if any(not neighbors[v] & support for v in VERTICES - support):
                continue
            supports += 1
            unseen = set(support)
            components = 0
            while unseen:
                components += 1
                frontier = [unseen.pop()]
                while frontier:
                    new_neighbors = neighbors[frontier.pop()] & unseen
                    unseen.difference_update(new_neighbors)
                    frontier.extend(new_neighbors)
            total += 2 ** (size - components)
    return supports, total


def expect_rejection(clique, bases, reason):
    try:
        check_witness(clique, bases)
    except ValueError as error:
        if str(error) == reason:
            return
        raise
    raise RuntimeError(f"broken control passed: {reason}")


def main():
    # These are the four lines of F3². Build witnesses from the two disjoint
    # edge factors, without selecting clique points by dominating support.
    directions = ((1, 0), (0, 1), (1, 1), (1, 2))
    clique = []
    bases = []
    for left, right in product(directions, repeat=2):
        clique.append(left + right + (0,))
        bases.append((left + (0, 0, 0), (0, 0) + right + (0,), (0, 0, 0, 0, 1)))
    check_witness(clique, bases)

    # Using addition in place of subtraction over F3 fails on this pair.
    if not orthogonal((1, 1, 0, 0, 0), (2, 2, 0, 0, 0)):
        raise RuntimeError("F3 determinant sign control failed")

    supports, weight = weighted_support_count()
    if (supports, weight) != (9, len(clique)):
        raise RuntimeError("weighted support count differs from literal witnesses")

    broken_clique = clique.copy()
    broken_clique[1] = clique[0][:-1] + (1,)
    expect_rejection(broken_clique, bases, "orthogonal pair in clique")
    expect_rejection(clique, bases[:-1] + bases[:1], "cover misses field vectors")

    print(f"PASS F3: {len(VECTOR_SPACE)} vectors, clique={len(clique)}, cover={len(bases)}")
    print(f"PASS support formula: {supports} dominating supports, weighted total={weight}")
    print("PASS determinant sign; rejected non-clique and incomplete cover")


if __name__ == "__main__":
    main()
