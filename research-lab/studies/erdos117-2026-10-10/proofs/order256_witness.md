# Exact order-256 example with a(G)=omega(G)=14

**Claim E117-EX-20261010. Status: EXACT_FINITE_CERTIFICATE; literature PRIORITY_UNCHECKED; no Lean formalization.**

Take V=F₂⁵ and W=F₂³. Order the ten wedge coordinates (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4). Define the three scalar components of an alternating B:V×V→W by the ten-bit integer masks **201, 756, 259**. Realize B by the bilinear cocycle in the accompanying [proof](line_packing_matching.md). The three scalar forms are linearly independent, rad(B)=0, and there is no B-isotropic 3-space. Thus the associated class-two group has order 2⁸=256 with |Z(G)|=|G′|=8 and G/Z(G)=V.

Noncommuting clique witness, 14 nonzero cosets:
3, 7, 13, 14, 15, 21, 23, 24, 26, 27, 28, 29, 30, 31.

Abelian-cover certificate, 14 commuting color classes:
[1,22,23]; [2,12,14]; [3,8,11]; [4,17,21]; [5,25,28]; [6,9,15]; [7,19,20]; [10,30]; [13]; [16,27]; [18,26]; [24]; [29]; [31].

The clique gives omega≥14 and a≥14. The coloring gives a≤14, so a=14. These witnesses alone do not upper-bound omega. The separate [C++ checker](../src/checker.cpp) reconstructs B from coordinate bits, checks all 31 points, independently enumerates every maximal clique using Bron–Kerbosch pivoting, and confirms omega=14 (5,328 maximal cliques). Therefore **a(G)=omega(G)=14**. Replay the checker from this study directory with g++ -std=c++17 -O2 src/checker.cpp -o checker, followed by timeout 35s ./checker 201 756 259. It uses the first raw record of [search.jsonl](../results/search.jsonl). Local independent replay exited 0 and took 3.66 seconds including compilation. GitHub CI for this new commit must be inspected separately.

This single finite case does not decide whether some other rank-five, three-coordinate pencil has a chromatic gap. It provides no all-group improvement to h(n), no uniform constant C and no eventual threshold n₀.
