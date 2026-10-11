# Rank-five three-coordinate alternating-map orbit census (2026-10-11)

**Exact scope:** All 3-dimensional F2-subspaces P of Alt(F2^5), a space with dimension ten. This is a completed *single-implementation* exact-arithmetic census. The full independent checker and explicit vertex-witness export remain pending. Neither a Lean verification nor a literature-priority claim is made.

## Claim contract

For each P, make a 31-vertex graph on nonzero x in F2^5, with an edge xy iff at least one scalar form in P has b(x,y)=1. A counterexample is any P with chi(Gamma(P))>omega(Gamma(P)). The competing hypothesis was that the rank<=2 equality was exceptional. The run tests the complete rank=3 scalar-pencil domain, not arbitrary groups.

## Proof method and receipt

An alternating form is determined by ten binary entries b(e_i,e_j), i<j. Each three-dimensional subspace has a unique binary 3x10 reduced row echelon basis. Iterating every free entry gives the Gaussian count [10 choose 3]_2 = 6,347,715. Four adjacent basis swaps and the elementary transvection e0 -> e0+e1 generate GL(5,2). For each plane the program constructs its orbit under induced pullbacks on scalar forms and normalizes to RREF; orbit membership is exact and basis independent.

The executed program found 22 orbits with sizes summing to 6,347,715. It evaluated a representative of each orbit by building the commutation graph using exterior-product parity, an exact Bron-Kerbosch maximum clique branch-and-bound, and an explicit proper DSATUR greedy coloring. In every orbit, the greedy coloring used exactly omega colors. An exhibited coloring proves chi<=omega; the clique proves chi>=omega. Each generator passed involution checks on all 1,024 scalar forms. The complete machine receipt is in results/worker2.log.

Because GL(5,2) relabels vectors, all graphs in an orbit are isomorphic. There were 16 zero-common-radical orbit types and six with nonzero common radical. The complete census finished within the 35-second cap, in about 1.11s with maximum RSS 139,264KiB. An earlier deterministic sample worker checked 1,000 faithful rank-three triples in 0.08s. Both exits were zero. The second implementation was a different *sampling workflow*, not an independent full-orbit certificate.

## Conditional group implication

If G/Z(G) is elementary abelian over F2, then G is class two and G' is elementary abelian. Commutation descends to an alternating bilinear B on its actual central quotient. A graph-independent set spans an isotropic subspace, whose full preimage is abelian. Conversely an abelian cover gives a proper coloring. Thus a(G)=chi(Gamma) and omega(G)=omega(Gamma), also for groups with infinite center. Combine the rank-three dimension-five census with the project's earlier dimension<=4 and two-output dimension-five checks to obtain the proposed finite-class statement: for dim_F2 G/Z(G)<=5 and dim_F2 G'<=3, a(G)=omega(G). This is conditional on accepting the one-method full census and those earlier checked certificates. It is NOT an improvement for h(n) over all groups.

## Next falsifier and status

An independent checker must reconstruct all 22 orbit bases, verify orbit sizes and generate explicit maximum cliques and omega-colorings by different algorithms. A mismatch suspends this class result. Source novelty is PRIORITY_UNCHECKED, validity is an exact one-method finite computational result, and full independent verification is PENDING.

## Reproduce

Run: g++ -std=c++17 -O3 src/orbits.cpp -o orbits ; timeout 35s ./orbits . Acceptance: exit 0, total=6347715, orbits=22, gaps=0, unknown=0 and equality of omega and greedy for every orbit. Program is copied from the exact locally executed source, but GitHub CI currently does not specifically replay this new census.
