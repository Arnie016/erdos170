# Actual execution receipt | 2026-10-28

Predeclared exact finite domains: every unordered pair of incident new edges between absolute vertices in prime-field orthogonal polarity graphs q=3 and q=5, and every original H-edge deletion. Domain size (q+1)*C(q,2)*q(q+1)^2/2: 288 for q=3 and 5400 for q=5. Seed none. Both source checkers independently reconstruct H, check vertex count, edge count and independence of absolute points. Code A counts noninduced C4 via half the sum of binomial(common-neighbors,2); code B enumerates 3 cycle patterns per 4-set. Acceptance frozen as exact full-domain min 2q-3, identical histogram and an attained minimum.

Worker A: compile g++ -std=c++20 -O3 -Wall -Wextra src/worker_a.cpp; run using timeout --signal=TERM --kill-after=1s 35s sh -c 'compile && exec binary'. Exit 0, wall 1.48s including compile.
Worker B: same compiler and timeout, worker_b.cpp. Exit 0, wall 1.12s including compile.

**Both**: q=3: 288 tested, min=3, equality=48, histogram {3:48,4:72,5:168}. q=5: 5400 tested, min=7, equality=360, histogram {7:360,8:840,9:4200}. Worker A separately checks that prior to deletion there are exactly 2q-1 cycles (negative control).

Compilers emitted only misleading-indentation formatting warnings in final output loops. No timeouts or failed workers. SHA256 (local identical source files):
common.hpp 20d31e41ff025e7b7b5689c924134ef0a3b7b5daff41614230fc957ffd54b6ad
worker_a.cpp 53dcdae9b1e01e87bc86364eab103a5e70dd130f30ac478a8bf8a307d85b3c08
worker_b.cpp c26ffa81460f5ce3c1bf63c7945dbdcdf904bea4adf243ab7a1677b880c18079

Finite checks do not establish all q; the written proof does. No new-study CI or Lean validation claimed in this report. No outreach or external compute.
