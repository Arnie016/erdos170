# Run receipt, 20 October 2026

**Claim contract.** Existence of a 39-element A subset {0,...,500}, with endpoints 0,500, realizing each distance 1..500 as b-a for a,b in A. Acceptance fixed before search: exact ordered marks, all 500 positive differences, an independent integer checker, explicit witnesses and positive/negative controls. Falsifier: any missing d. Stop: first independently verified 39-mark completion.

**Selection.** A #242 fixed-first-denominator claim was considered but a materially similar observation was already known in a 2026 preprint (arXiv:2608.24035). Chose #170 because the Oct18 exact two-mark exclusion closed only one older seed, leaving a structurally different construction possible.

**Worker 1** ran once, sequentially, `timeout -k 1s 33s bash -c 'g++ -O3 -std=c++17 -Wall template_search.cpp -o template_search && exec ./template_search'`, exit 0; outer wall 1.562s, inner C++ search 0.0290308s. Predeclared r=4..8 Wichmann single/two-gap domain with optional seeded three-gap sampling. The 30,350th evaluated candidate was complete at r=5 after shortening one 23-gap. The random phase was not reached. **Raw logging caveat:** `exact_evaluations:0` in raw_result.json is a logging bug caused by goto skipping assignment; `total_evaluations:30350` and raw_stderr BEST 0 at trial 30350 are authoritative. Raw files remain unchanged.

**Worker 2** ran once, `timeout -k 1s 33s python3 independent_verify.py`, exit 0; outer wall 0.998s, inner 0.001198s. Checks 39 exact marks, 500 positive distances via integer bitset, 500 separate witness pairs, old 40-mark positive control and delete-262 negative control. All assertions passed. `witnesses.csv` SHA256 d77200b644c1c9ccbf2d206d8b8f02b5ef9233dc281da238d849a8886606e0a1.

Two experimental workers total (2/2). No GPU, paid API, Mac or background run. No mathematics outreach sent. The generated independent evidence proves F(500)<=39, not optimality, asymptotic improvement or literature novelty. Catalog registration must wait for an authentic evidence commit/CI receipt rather than inventing them.
