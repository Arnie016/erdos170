# Frozen acceptance contract: SIDON-3BIN-FULLSCALAR-2026-11-20

Question: does the 19 November exact rational two-kernel witness beat **all** feasible scalar symmetric length-three Hou–Zhao systems, t∈[0,1/2], not merely t<1/3?

Domain: m=3, L=1; scalar symmetric p=(t,1−2t,t), arbitrary real w0,w1,w2, default wj=1 for j≥3. Four q covers, b>0; mixed rational witness as in PROOF.md.

Acceptance frozen before experiments: piecewise KKT-optimal scalar boundary weights; all real endpoints and q constraints; rational lower bound strictly above the mixed ab; two workers maximum, each ≤35 seconds. Worker 1 does exact symbolic factorization and negative control; worker 2 independently audits all 190 (k,l) finite grid pairs via exact rational active sets. A grid minimum does not establish a continuous optimum.

Competing explanation: a scalar t∈[1/3,1/2] beats the mixture. Falsifier: a feasible scalar with ab≤1150153/1253320 or an incorrect sign/KKT/b positivity argument. Stop after two workers or any unrepaired conflicting certificate. Parent implication is only a structural separation inside a known vector-smoothing theorem, not improved Sidon bounds.

Primary source: Hou–Zhao arXiv:2607.01169v3, 4 September 2026, theorem 2.1 and equations 2–4.
