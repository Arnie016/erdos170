# Run and provenance | 2026-10-17

Selected question: n=7 t=5 modular distinct subset sums, complete classification modulo 133. Considered #242 first; new sieve work arXiv:2608.24035 makes a small fixed-offset scan less discriminating. Prior exact n=6 census in this repository was read but not replayed.

**Claim contract:** All increasing canonical seven-subsets of {1,...,66}; each must have 128 distinct sums modulo133. Falsifier: one admissible set outside the three unit/sign orbits or with maximum dyadic overlap <=5. Acceptance: exact complete search by two mathematically different recurrences, direct independently rebuilt subset sums for all accepted leaves, signed-unit orbit classification. Stop after two <=35-second workers.

**Worker 1:** Python 3.13.5, command timeout --signal=TERM --kill-after=2s 35s python3 search_mod133.py search_output.json. Exit 0, program 11.701761413 seconds. Nodes [1,60,1830,36656,490780,3322318,2942381,162], first_bad=null.

**Worker 2:** g++ 14.2.0 C++20. Command timeout --signal=TERM --kill-after=2s 35s bash -c 'g++ -std=c++20 -O2 -Wall -Wextra check_mod133.cpp -o check_mod133 && ./check_mod133 independent_output.json'. Exit 0, program 0.447057 seconds plus compilation. Identical depth counts, 3 orbits of size54, dyadic max-overlap histogram 6:108 and 7:54. Negative control has 29 distinct subset sums. Both workers stopped normally. No paid services, GPUs, Mac jobs or outreach.

**Local verification hashes SHA256:**
- search_mod133.py e53b363cd55d31354eb3087daf7df4007579141770c058afd1d69f4485980f1b
- check_mod133.cpp 24f20d494a41701165eddbac215259f2ac5bd78b2e537615a0aa13666966e73c
- search_output.json 462f838af8f7f02a51a2a2569c39bdf259109a408e4626bb5802b99f79e128d5
- independent_output.json 8e8828d3a53abb062b7d3f9c62b2cfe5f346194968ddfd42c5d0be7afe543c6a

Public source: Cambie–Gao–Kim–Liu, arXiv:2308.03748 (2023; Acta Arithmetica 2025), Introduction Theorem1.5; Sections7 and Proposition A.1 read. Templates at t=5 known; universal completeness only conjectural. #242 arXiv:2608.24035 abstract read, not used in mathematical proof. This repository's main HEAD e341d5a244ce1632841e6da078914e9df6d9b469 was read before work. A prior relevant private Gmail reply was inspected but not published or used as a proof.

**Publication scope:** The exact source and proof are released for review; catalogue promotion will await CI for this precise revision. No claim of a new universal theorem, formal Lean proof or literature priority.
