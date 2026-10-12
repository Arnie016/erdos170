# Checkpoint: 2026-10-12 rank-incidence identity

Previous current remote head was 79bf49d31caedac7e2132e5a2377bf04a2e7a297. Files read on that head: research-lab/STATUS.md, AGENTS.md, README.md, SOURCES.md and full proof files for 2026-10-09 rank-five two-output census, 2026-10-08 rank-four classification and 2026-10-06 order-64 witnesses. Older unrecovered private packages are not promoted to evidence.

**Established:** for spanning and faithful alternating B:F₂⁵×F₂⁵→F₂³, the basis-invariant scalar-radical incidence double count gives 3n₁+n₂=7+6t, so n₂≡1 (mod 3). Validity ANALYTICAL_PROOF; priority unchecked; Lean UNFORMALIZED. 1,007 faithful SHA-fixed triple regressions passed. This is not a global a(G) or h(n) bound and not a chromatic-gap example.

**Next active question:** Does any rank-five, three-coordinate faithful alternating map have χ(B)>ω(B)? A legitimate B with an independently verified upper bound on ω(B) and strictly larger lower bound on its isotropic-subspace covering number is a falsifier of equality. Stratify exact cases by t and contraction rank rather than repeating random sampling; stop if only empirical counts change. **Parked:** Lean group-to-graph reduction.

**Public sources:** Erdős #117 https://www.erdosproblems.com/117 (fetch returned 403 this run); Guillaume Lecomte, “Sharp Asymptotics for Abelian Covers of Groups with Bounded Noncommutativity,” arXiv:2608.20507v1 (2026-08-20), abstract read, proof not read. Privately reported later work is not imported or published.

**Execution:** two sequential local-container Python workers, no GPU; 35-second caps; each exit 0; no ongoing worker; no outbound messages. Reproduce with `timeout -k 2s 35s python3 src/worker1.py` then worker2. Sample terminal outputs are stored in results/.
