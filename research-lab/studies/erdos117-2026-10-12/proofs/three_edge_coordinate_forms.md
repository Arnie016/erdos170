# Three-edge binary coordinate maps have equal cover and clique numbers

**12 October 2026 | ANALYTICAL_PROOF + EXACT_FINITE_CERTIFICATE | PRIORITY_UNCHECKED | UNFORMALIZED**

Let H be any simple graph with three edges on five labelled vertices. Set V=F_2^5 and W=F_2^{E(H)}, with B_H(x,y)_{ij}=x_i y_j+x_j y_i over F_2. Define omega(B_H) as the maximum number of pairwise B_H-nonorthogonal vectors and a(B_H) as the minimum number of B_H-isotropic subspaces covering V.

**Theorem.** For every such H, a(B_H)=omega(B_H). By vertex relabelling the four cases are: triangle plus two isolates (7), star plus one isolate (9), four-vertex path plus one isolate (9), and three-vertex path disjoint from one edge (15). The 120 labelled graphs split into these classes of sizes 10,20,60,30.

**Group legitimacy.** On V x W use (x,z)(y,t)=(x+y,z+t+f(x,y)), where f(x,y)_{ij}=x_i y_j for i<j. Bilinearity proves associativity; the commutator is B_H. Commutators at the edges span W. The common radical is the span of isolated vertices, since testing B_H(x,e_i) detects each incident endpoint coefficient. Hence Z(G_H)=rad(B_H) x W, and only the disjoint-path-plus-edge case gives an actual five-dimensional central quotient. The group has order 256. Mutually commuting vectors span isotropic spaces, whose preimages are abelian. Covers correspond to colorings of the noncommuting graph. A k-clique gives a lower bound, while k isotropic covering subspaces give an upper bound.

**Triangle:** after discarding isolated coordinates, all seven nonzero F_2^3 vectors pairwise noncommute. Their seven lines cover.

**Star:** write central bit t and three leaf bits u. The eight (1,u) are pairwise noncommuting, and every nonzero (0,u) noncommutes with all eight. Eight corresponding lines plus the hyperplane t=0 form a nine-subspace cover.

**Four-vertex path:** with edges 01,12,23 and integer encoding x0+2x1+4x2+8x3, the nine-clique is {5,6,7,9,10,11,13,14,15}. Nine pairwise commuting classes partition the nonzero vectors:
{6}; {7}; {14}; {15}; {2,8,10}; {3,11}; {1,4,5}; {9}; {12,13}.
Each class spans an isotropic subspace. The unused fifth coordinate is global radical.

**Three-vertex path disjoint edge:** these are separate commutator blocks with independent central coordinates. The three-vertex path has a=omega=5: four middle-bit-one vectors form a clique, any nonzero middle-bit-zero vector extends it, and four lines plus the middle-bit-zero plane cover. The two-vertex edge has a=omega=3. Cartesian products of the cliques and direct sums of the cover subspaces give a=omega=15. This uses the elementary tight-factor product inequality and claims no novelty for it.

A three-edge graph either contains a triangle, is a connected three-edge tree plus an isolated vertex (star or path), or consists of paths on three and two vertices. These four cases exhaust the domain.

**Exact certificate.** The checked raw_witnesses.jsonl gives four complete cliques and all 31 color assignments. Run src/independent_check.py from this directory with Python 3.10+. It independently constructs symmetric matrices for B, checks every clique pair, every same-color pair and each class span. It also verifies all 120 labelled edge sets are accounted for by these four permutation orbits and rejects two intentionally corrupted witnesses. Observed local output: INDEPENDENT_CHECK_PASS; Python 3.13.5; 1.29 s; exit 0. The locally executed C++ generator completed all four cases in <0.01 s (timer resolution), exit 0. Both workers had 35 s hard limits. No Lean check was run.

**First unsupported step:** an arbitrary three-dimensional space of alternating forms on F_2^5 need not admit this special coordinate representation. This result does not improve h(n) over all groups, any global constant or any finite exact-formula threshold.

**Next falsifier:** find an indecomposable, common-radical-free three-output form with chi>omega, certified by a clique lower bound and an independently checked obstruction to omega-colorability. A sampled failure to find one proves nothing universal.

Public background: https://www.erdosproblems.com/117 and Guillaume Lecomte, arXiv:2608.20507v1. No unpublished text or private correspondence enters this proof.
