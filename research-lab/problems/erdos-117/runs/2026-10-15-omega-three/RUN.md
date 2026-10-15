# 2026-10-15 | all-group first threshold

**Question:** Does omega(G)<=3 force a(G)<=3, without finiteness assumptions? **Verdict:** ANALYTICAL_PROOF. Full proof in PROOF.md. Literature priority not asserted; no kernel formalization.

**Acceptance:** Derive a three-centralizer cover, show the intersection must be central by an explicit five-element obstruction, prove a cover of size three and exhibit D8 as sharp. Falsifier: omega=3 but center quotient not C2 x C2 or a>3.

**CPU receipt:** one worker, Python 3.13.5. Exact domain D8,Q8,S3,C4. Generated full Cayley tables and checked associativity and inverses, counted cliques by exhaustive subsets and covers by enumerated abelian subgroups. Exact expected triples (omega,a,|G/Z|) were frozen before running. All four passed, exit 0, 0.883 s external wall, subprocess hard timeout 10 s. Negative controls S3 and C4. No GPU or second worker. The independently executed local script is reproduced here, not a formal proof. No GitHub CI verdict is asserted until this PR is checked.

**Literature:** public arXiv:2608.20507v1 (20 Aug 2026), theorem 2.2 read in full; historical Scorza three-subgroup theorem cited only for context. This finite-threshold proof has no imported theorem dependencies. Current status of erdosproblems.com/117 unavailable due to HTTP 403.

**Limits:** Does not prove h(4), a uniform constant, eventual exact threshold or global asymptotics. Do not present this as a newly discovered theorem.

**Next:** Test h(4)=4; a certified G with omega<=4 and a>=5 refutes it. Park the rank-five three-form gap question rather than running two active targets.