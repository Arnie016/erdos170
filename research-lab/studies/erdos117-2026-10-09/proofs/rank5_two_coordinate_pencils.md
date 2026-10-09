# Rank-five two-coordinate commutator pencils

Status: exact finite certificate plus ordinary group reduction. Literature priority unchecked. The census is computational, not yet a Lean kernel proof.

## Theorem

If G/Z(G) is elementary abelian over F2 with dimension at most five and dim_F2 G'<=2, then a(G)=omega(G), including groups with infinite center.

The new computation handles every two-dimensional scalar pencil on V=F2^5. The lower-dimensional result is supplied by the adjacent rank-four study.

## Group reduction

Since V is elementary abelian, G has class at most two and [g,h]^2=[g^2,h]=1. Hence the commutator is alternating bilinear B:V x V->G'. Noncommuting sets are graph cliques. Commuting color classes span B-isotropic subspaces whose preimages are abelian. Conversely an abelian cover colors the graph. Thus omega(G)=omega(B) and a(G)=chi(B).

For actual dim V=5, a zero derived group is impossible. A one-dimensional derived group gives a single alternating form on an odd-dimensional space, which has nonzero radical; this contradicts faithfulness of the actual central quotient. For formal nonfaithful maps one may factor out the common radical instead. The remaining case is dim G'=2, hence two independent scalar forms.

## Exhaustive domain

Alt(F2^5) has dimension ten. A scalar plane consists of {0,a,b,a+b}; distinct bases give the same commuting relation. There are exactly

`((2^10-1)(2^10-2))/6=174251`

planes.

Worker 1 enumerates their RREF bases, constructs the 31-vertex graph using parity of ten wedge coordinates, computes maximum clique by coloring-bound branch-and-bound and checks omega-colorability with exact DSATUR recursion. Worker 2 enumerates sorted triples a<b<a XOR b independently, rebuilds the graph, recomputes maximum clique with Bron-Kerbosch, enumerates all 374 vector subspaces by span closure and checks a cover by omega maximal isotropic subspaces using exact branching. Every canonical pencil key and clique number is compared.

| omega=a | pencils |
|---:|---:|
|5|2,821|
|6|6,510|
|9|112,840|
|11|52,080|
|Total|174,251|

All planes have matching covers. The domain intentionally includes nonfaithful planes, which does not omit any actual central quotient.

## Reproduce

From this directory:

```sh
mkdir -p .build results
g++ -std=c++17 -O3 src/rank5_two_pencils.cpp -o .build/first
g++ -std=c++17 -O3 src/worker2_iso_cover.cpp -o .build/second
timeout 35s .build/first
timeout 35s .build/second
```

Worker 1 returning exit zero alone is not acceptance: check status FULL_NO_GAP and count 174251. Worker 2 must report INDEPENDENT_PASS and the same count. The repository reproduction driver checks these exact values. The raw 7.8 MB JSONL file is regenerated and included in CI artifacts rather than trusted as opaque solver output.

## Boundaries

Nothing here asserts no gap for output dimension >=3 in rank five, or for rank six. In rank six a nondegenerate binary symplectic form gives omega=7 and a=9. The clique upper bound follows from the rank of the off-diagonal-ones Gram matrix; seven points exist. Every isotropic subspace has at most seven nonzero vectors, so at least nine are needed to cover 63 points. A symplectic spread of nine 3-spaces supplies the upper cover.

For the computed class a=omega, the normalized ratio cannot beat the D8 benchmark 3/(2sqrt(2)). This is a finite-class exclusion, not an optimal uniform constant or a global h(n) theorem.

## Open question

Does any three-dimensional scalar pencil on F2^5 have chi>omega? A single explicitly checked graph and incompatible clique/cover values would settle this existence question. Failure of a bounded search is not a negative answer.
