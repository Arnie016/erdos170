# Local two-mark rigidity audit, N=500

This dated original finite study uses two distinct compiled C++ algorithms and the exact 39-mark baseline from the archived 2026-10-16 one-edit audit. See PROOF.md for the exhaustive-domain reduction and RUN_REPORT.md for the actual process receipts. Both workers hardcode the positions displayed in candidate.json. Their executable source is reproducible with g++ 14.2+, -std=c++20 -O3.

Compile each worker with `g++ -O3 -std=c++20 -o worker_a src/worker_a.cpp` and similarly for worker_b. Run sequentially with a <=35-second hard timeout. Expect 2,437,560 two-edit candidates, 0 complete and minimum 2 missing in each; only worker A also checks the 17,094 single-edit cases.

This run is not a claim of global sparse-ruler optimality or a certified asymptotic advance. The repository catalog requires a current CI evidence_run for exact finite studies; do not register before that receipt.
