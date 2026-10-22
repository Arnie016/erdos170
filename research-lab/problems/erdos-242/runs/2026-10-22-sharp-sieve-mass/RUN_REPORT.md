# Mathematics portfolio research report, 2026-10-22

**Selected:** Erdős #242 auxiliary problem: evaluate Dahan's W(J), a coprime totient-weighted double sum. Chosen over repeating #170 repairs because the published paper explicitly identified an unresolved coefficient question that admits a rigorous analytic test.

**Outcome:** ANALYTICAL_PROOF, PRIORITY_UNCHECKED, UNFORMALIZED.
W(J)=9/(2π²)·log²J+O(log J), exact leading coefficient 0.45594532639052. Dahan's Lemma 4.22 had only a 0.1519817755 lower coefficient and Section 7 requested ≥0.45. This closes the mathematical coefficient calculation, subject to outside priority audit; no general Erdős–Straus theorem changed.

**Proof:** Exact two-variable Dirichlet factors, including p=2; separate ζ(1+s)ζ(1+t) and absolutely summable remainder H, evaluate H(0,0)=9/(2π²). Convolution with harmonic numbers gives the O(log J) error. See PROOF.md.

**Bounded verification:** One Python standard-library worker with external timeout 35s, actual wall 3.29s, exit code 0. Tests all 1,024 u,a≤32 for totient parity and all 256 u,a≤16 for independent rational convolution, with one negative coprimality control. Direct W(J)/log²J values at J=100,300,1000,3000 match Dahan printed ratios within 0.001. Raw stdout, timeout receipt, SHA256 and checker code included. No other worker or paid compute was used.

**Limitation:** Improved local lower mass alone is not a representation for 4/n, and improving the published exceptional-set exponent awaits an audit of the transfer hypotheses. Full literature precedence not verified. A missing canonical-site fetch is not proof of open status.

**Continuity:** Main HEAD e92a58695dcb638c9f6773b23f4653590a329504, prior #242 research branch 3ebda32a8c19a5b4fccbe7dc9c95bf2aee08b665, prior six catalog records and active #117 PR #3 were inspected. No related inbound math correspondence; no email sent. No schedules changed.

**Next test:** verify the coefficient's precise propagation in Dahan Lemma 4.23 and Corollary 4.31 or isolate an obstruction.
