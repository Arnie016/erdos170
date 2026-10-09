# Every binary central quotient of dimension at most four has a=omega

Status: exact finite certificate plus ordinary group-to-linear reduction. Literature priority unchecked. Not a Lean formalization of the census.

## Theorem and reduction

Let G be any group such that G/Z(G) is elementary abelian over F2 with dimension at most four. Then a(G)=omega(G). G may have an infinite center.

Since the quotient is abelian, commutators are central. As g^2 is central, [g,h]^2=[g^2,h]=1. The derived subgroup is elementary abelian, and the commutator gives an alternating bilinear map B:V x V->W.

Two cosets commute exactly when B(v,w)=0. A color class in the noncommuting graph spans an isotropic subspace; its full preimage is abelian. Conversely abelian subgroup covers color the graph. Thus omega(G)=clique(B) and a(G)=chromatic(B). Lower dimensions extend by zero directions to dimension four; independent twins and isolated radical points do not change the invariants. The abelian case has both invariants one.

## Complete finite domain

Every alternating map factors through the exterior square: B(v,w)=L(v wedge w). Since dim Lambda^2(F2^4)=6, the commuting relation is determined exactly by K=ker L<=F2^6. Every such K is realized by the quotient map. There are

`1+63+651+1395+651+63+1=2825`

subspaces, the sum of the binary Gaussian binomial coefficients.

Worker 1 enumerates unique RREF bases, constructs graphs using exterior products, computes exact maximum cliques and exact DSATUR colorings. Worker 2 independently builds all kernels by span closure, constructs scalar annihilators, evaluates forms directly, computes clique sizes by subset DP and solves the minimum cover by actual isotropic subspaces of V. It compares every kernel identifier and pair of invariants with worker 1.

| omega=a | kernels |
|---:|---:|
|1|1|
|3|35|
|5|189|
|6|210|
|7|15|
|9|1,485|
|11|490|
|13|315|
|15|85|
|Total|2,825|

All values match. Exhaustion proves this finite assertion, combined with the reduction above, not any assertion in larger dimension.

## Reproduction and boundary

Run `python3 src/worker1_rref_graph.py`, then `python3 src/worker2_bfs_subspace_cover.py`. Both full JSONL lists are regenerated in results/. The second checker uses all 67 quotient subspaces and neither the first enumerator nor its coloring algorithm. This is implementation independence, not independent human peer review.

Kernel subspaces classify commuting relations, not necessarily group isomorphism classes. Including degenerate maps is an intentional superset. No global h(n) improvement follows.

The normalized cover ratio in this class is n/2^(n/2)<=3/(2sqrt(2)). Thus no member improves the uniform-constant lower benchmark. A chromatic gap for an elementary binary central quotient must occur beyond dimension four.
