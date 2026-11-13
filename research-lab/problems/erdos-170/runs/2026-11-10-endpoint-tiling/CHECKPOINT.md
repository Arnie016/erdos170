# Checkpoint | 2026-11-10

**One question:** Under a tight pair budget, do two end tails necessarily cover the largest distances of a restricted sparse ruler?

**Answer:** No. If $L=\{0,a\}$ and the reflected right tail contains $q$ marks while $2q$ longest distances must be covered, exact tiling forces $a\mid q$. The right tail is uniquely the union of alternating blocks of length $a$. For $q=6$, exactly four of 5,082 endpoint profiles cover all twelve longest distances.

**Parent implication:** Local branch pruning and a one-mark stronger right-tail lower bound whenever $a\nmid q$. No improvement to the asymptotic Erdős #170 limit, to the 39-mark length-500 question or to any global $F(N)$ bound.

**Evidence:** PROOF.md, worker_pairs.py, worker_masks.py, RUN_RECEIPTS.json. Ordinary proof, not Lean. Public claims exclude private correspondence.

**Next discriminating test:** Recover an actual saved incomplete length-500 ruler with its one-replacement exclusion, then determine whether endpoint tiling can eliminate a *specified* class of coordinated two-replacement repairs. If no meaningful cut results, park this technique and rotate problems.
