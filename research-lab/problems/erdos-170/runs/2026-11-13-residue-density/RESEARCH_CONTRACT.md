# Frozen research contract

**Selected question.** Does the conditional local bound \(M(r,t,a)=\sum_s\lceil m_s/r\rceil\), derived 12 November 2026, survive corrected independent finite verification without weakening its exact acceptance test?

**Domain.** All integers \(r\ge2,a\ge1,t\ge1,N\ge2t-1\) with \((r-1)a<t\), exact left window \(\{0,a,\ldots,(r-1)a\}\). Finite verification: \(r=3,t=30,1\le a\le14\); second independent exhaustive domain \(r=3,t=12,1\le a\le5\), all 4096 right-tail masks per spacing.

**Dependencies.** Ordinary residue-chain lower-bound proof and explicit construction. The 12 November initial checker failed an incorrect negative control requiring \(M(3,12,4)>4\). Correction: \(M(3,12,4)=4\), and \(M(3,12,5)=5\).

**Falsifier.** Any right tail smaller than the formula which covers the fixed offset domain, or any disagreement between constructive and independent results. No unstated filtering of right-tail masks.

**Frozen acceptance.** Both workers exit 0 within separate 35-second hard limits, reconstruct the same 14 values [10,10,12,12,10,12,14,14,12,10,11,12,13,14], and check their negative controls. Independent exhaustive values for \(t=12,a=1,\ldots,5\) are [4,4,6,4,5]. Acceptance script checks all values again.

**Parent implication.** Prunes a specified endpoint family in sparse-ruler construction search. It is not a global F(500) lower bound, an optimum certificate or an asymptotic Erdős #170 advance.

**Stop.** At most two sequential worker processes, each <=35 seconds, no retries to chase success; once passed, publication in the already-open PR with a dedicated CI job, not a new branch.
