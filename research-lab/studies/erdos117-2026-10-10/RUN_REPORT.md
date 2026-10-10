# Dated research receipt: 10 October 2026

The current public basis read before choosing work was main e103025a88c0cec188dd94a2a74b89819f40487a, with STATUS.md, AGENTS.md, the 2026-10-09 proof and its first worker source. The old round27 package was unavailable in targeted private Library searches, so it was not imported.

Primary source audit: Lecomte, arXiv:2608.20507v1 (20 August 2026), PDF pages 1-6 extracted and pages 3 and 6 visually inspected, including the exact all-group finite-reduction Lemma 2.1 and the asymptotic Theorem 2.2. The public theorem asserts log₂ h(n)=n/2+O(√n(log(n+2))³); no public eventual exact formula was verified. Erdős #117 site was last edited 23 January 2026, earlier than this public preprint.

Predeclared experiment: up to 20 fixed-seed pseudorandom triples of distinct linearly independent ten-bit alternating forms, stop at eight faithful pencils with no B-isotropic 3-space, or on first gap. Negative control is the all-zero map, which must possess an isotropic 3-space. The explicit falsifier for an alleged gap is an omega-clique plus an omega-coloring. Sampling with replacement is NOT an isomorphism census.

Worker 1 C++17 O2 with xorshift32 seed 20261010: 8 attempts, 8 qualified; exact maximum-clique and DSATUR both returned ω=χ, values 14, 15, 17, 13, 13, 15, 13, 13. Negative control passed. Hard timeout 35 s; wall 1.39 s including g++ compilation; exit 0. Local generator and all eight raw records are in the separate dated working archive, not represented here as a global theorem.

Worker 2 source src/checker.cpp was fixed before worker 1. Direct-coordinate bilinear evaluation, zero-global-radical check, exact clique bound by Bron–Kerbosch and cover witness check. It independently tested the first triple (201,756,259) only. Hard timeout 35 s; wall 3.66 s including compilation; exit 0. Exact stdout: INDEPENDENT_CERTIFICATE_PASS omega=chi=14 maximal_cliques=5328 31_vertices=yes nonzero_radical=0 no_isotropic_3space=yes. Only this first sample is promoted to an independently replayed finite example.

Two and only two local experiment workers executed, neither timed out, neither was restarted. No private correspondence, drafts, contact ledgers, third-party manuscripts or local personal filesystem paths are included. No email was sent. General Erdős #117 and #170 remain untouched at the level of global bounds.
