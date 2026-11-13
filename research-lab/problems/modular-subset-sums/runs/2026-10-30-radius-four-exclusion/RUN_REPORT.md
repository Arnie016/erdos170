# Run report | 2026-10-30

Frozen domain: all 460 units modulo 517; graph edges require exact 128-point B_a intersection and four disjoint translates covering the 512 represented residues. Positive control centered {0,±1,±3}: 156 first-stage units, 308 pairs, maximum clique 9. No random seed.

Worker A: C++20 compiled with g++ -O2, then executed under one 35-second hard timeout, exit 0, externally measured wall 3.978324112 seconds including compilation. Two harmless misleading-indentation warnings. SHA256 of locally executed source 8bde58b3926ae1c4baf0d27611c681b896e7b1ca7c072c7d5d07bb2857e235b1.

Worker B: CPython, independent binary recurrence, set checker and maximal-clique search. 35-second hard timeout; exit 0, externally measured wall 1.447802919 seconds. SHA256 of locally executed source 547c2db461b89d8e3bc5b7a1cbdd55b70294eecc66f337e451f1405d6038dc33. B compared the entire sorted A vertex and edge lists and clique number, not only summary counts.

Both found 132 admissible first-stage candidates, 40 compatible pairs, maximum clique 2. The graph's 18+18 bipartition and 96 isolated vertices are recorded in PROOF.md / results/graph.json. Full local raw stdout, stderr (sanitized), input and output evidence are in the dated user-visible research package. GitHub Actions does not yet independently reproduce this new study; earlier green jobs are not evidence for this new checker. No emails sent; no schedule changes. Catalog promotion is pending new-study CI.
