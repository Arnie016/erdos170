# Run report, 2026-10-10 (public, sanitized)

**Fixed claim:** For all rank-three scalar subspaces of Alt(F₂⁵), decide whether their graphs satisfy chromatic number equal to clique number. An explicit graph with χ>ω is a falsifier. All-group h(n) is out of scope.

**Source checks:** Live arXiv:2608.20507 still listed v1 (20 Aug 2026). The Erdős problem website's direct fetch returned 403; a search-indexed version was historical, not an authoritative current status. Read the relevant 2026-10-08 and 2026-10-09 proof notes in this repository before extending their domain. No confidential communications or manuscripts are included.

**Worker 1:** Seeded distinct-sample screen of 2,048 independent triples, C++17 mt19937_64 seed=11720261010, hard timeout 32s; exit 0 in 0.05s; found no gap. Sample scope only. This screen is superseded by the next complete census.

**Worker 2:** Compiled src/full_three_orbits.cpp using g++ 14.2.0 with -std=c++17 -O3 -Wall -Wextra. Command: timeout -k 1s 32s ./full_three_orbits. Exit 0, wall 3.34s, max RSS 269040 KB. Complete RREF three-plane enumeration of size 6,347,715; GL5 orbit closure via five-cycle and transvection yielded 22 orbits; each representative passed exact clique/coloring comparison. Checks built graph edges with separate 5×5 matrix and wedge-product implementations and validated generator action on every input pair. See evidence/three_orbits.stdout. This step used CPU only, no paid compute, no Lean run.

**Limitation:** The independently written Python checker was not run, and a fully independent orbit enumerator remains to be implemented. Literature priority was not resolved. Earlier finite dependencies were read but not rerun; no global h(n) or uniform-constant consequence. No mathematical outreach sent.
