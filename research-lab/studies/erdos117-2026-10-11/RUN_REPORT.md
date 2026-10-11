# Actual local run, 2026-10-11

**Contract:** all three-dimensional binary scalar planes of Alt(F2^5); no counterexample would mean equality of graph clique and chromatic numbers in that finite domain. A single graph with chi>omega is the explicit falsifier. The competing hypothesis was that rank-three interaction produces a low-dimensional gap missed by rank-two studies.

Worker 1: `g++ -std=c++17 -O3 -Wall src/search.cpp -o search; timeout --signal=TERM --kill-after=2s 35s ./search`. Deterministic xorshift32 seed 0x1172026; first 1000 independent scalar triples without common radical; 1016 draws, no gap, no unknown. Exit 0, wall 0.08s, peak 1792 KiB. Raw result in results/worker1.log.

Worker 2: `g++ -std=c++17 -O3 src/orbits.cpp -o orbits; timeout --signal=TERM --kill-after=2s 35s ./orbits`. Exact RREF and GL5 orbit BFS, complete domain 6347715, 22 orbits, 16 faithful and 6 radical orbit types, all greedy coloring numbers equal to exact maximum clique. Involution and RREF smoke checks passed. Exit 0, wall 1.11s, peak 139264 KiB. Exact log in results/worker2.log. No workers left running.

**Limits:** Only one method completed the full census. The random sample is not an independent complete checker; raw vertex-level clique/cover witness arrays are not yet exported. No Lean was run. Current CI does not replay this new program, so any existing CI success is not an independent confirmation of these new numerical findings.

No email outreach was sent. No private correspondence, third-party unpublished manuscripts or credentials are published. The all-group bound h(n) remains unchanged.
