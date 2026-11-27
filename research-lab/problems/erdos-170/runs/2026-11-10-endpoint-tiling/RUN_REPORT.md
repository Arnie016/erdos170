# Execution and evidence report | 2026-11-10

**Frozen domain:** $N=23,t=12,q=6$; $a=1,\ldots,11$; $R=\{0\}\cup B$ with $B\subset\{1,\ldots,11\}$ and $|B|=5$. Total $11\binom{11}{5}=5,082$ profiles.

**Frozen acceptance:** Exactly four profiles, with $a\in\{1,2,3,6\}$ and the explicit right tails in PROOF.md. Both independent integer implementations must agree. The false-offset and damaged-certificate controls must fail as intended.

Two local Linux/Python 3.13.5 workers ran sequentially under separate 35-second hard timeouts. Observed wall times: **0.699031 s** for actual ruler differences and **0.764571 s** for integer generating-polynomial bitmasks. Both exited zero and neither timed out. Run in an isolated environment, not the user's Mac or Vanda. Execution worker source SHA-256:
- worker_pairs.py: c70e4d6d8cda823cb11830f76f9e01b0da03be4e6b11b361ecd9d93dca3d2a12
- worker_masks.py: 187e74b2c9e58a823d7ada692edf1ea4e2cdb39eddfcca8a71e2d442de562543

Published worker blob IDs match executed files. Transient local absolute paths and private correspondence are omitted.

**Remote CI:** Dedicated endpoint-tiling-q6 job is defined in .github/workflows/math-research-ci.yml on this PR. The proof's full quantifiers are not Lean-formalized; remote CI reproduces the exact finite experiment, not the universal polynomial argument.

**Scientific scope:** Only local tail-profile pruning; no new global sparse-ruler optimum or asymptotic bound.
