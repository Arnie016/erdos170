# Execution and evidence report | 2026-11-10

**Frozen domain:** (N=23,t=12,q=6), (a=1,dots,11), (R={0}cup B), (Bsubset{1,dots,11}) with (|B|=5): (11inom{11}{5}=5,082) profiles.

**Frozen acceptance:** exactly four profiles, (ain{1,2,3,6}), each with the explicit right tail in PROOF.md; both independently implemented verifiers agree; negative controls fail as intended. No probabilistic trials or numerical approximations.

Two local Linux/Python 3.13.5 worker processes ran sequentially under separate 35-second hard deadlines. Actual observed wall times: **0.699031 s** for direct actual-ruler differences and **0.764571 s** for generating-polynomial bitmasks. Both exit 0, neither timed out. Run from an isolated environment, not the user's Mac or Vanda. Worker source SHA-256 at execution: `c70e4d6d8cda823cb11830f76f9e01b0da03be4e6b11b361ecd9d93dca3d2a12` and `187e74b2c9e58a823d7ada692edf1ea4e2cdb39eddfcca8a71e2d442de562543`. The complete original receipt includes transient local paths; omitted from this public summary.

Remote CI status: **NOT YET OBSERVED for this study**. The existing historical CI workflow does not automatically execute these new checkers. `python3 run_workers.py` is a bounded, local reproducibility entry point. This result is not Lean-formalized and makes no global optimality claim.
