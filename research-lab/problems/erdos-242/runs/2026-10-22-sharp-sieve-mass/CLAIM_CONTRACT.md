# Frozen claim contract, 22 October 2026

**Statement and domain:** Let W(J)=Σ_{1≤u,a≤J; gcd(u,a)=1}1/φ(4ua). Prove for all sufficiently large integer J that W(J)=[9/(2π²)](log J)²+O(log J), with absolute implied constant. No claim about every 4/n decomposition.

**Dependencies:** Exact totient Euler factors, standard Euler products and harmonic estimates, absolute two-variable Dirichlet convolution. Competing explanation: the approximately 0.46 coefficient extrapolated from finite data is an artifact, or the even prime or coprimality is miscounted.

**Falsifier:** An incorrect local coefficient, divergence of the proposed h convolution, or a direct rational counterexample to its coefficient identity invalidates the proof. Acceptance requires an inspectable ordinary proof plus finite rational regression over every u,a≤32 (totient parity), every u,a≤16 (convolution), and four published numeric benchmarks within 0.001. At most one worker with a hard 35-second timeout. Stop once proved and independently checked.

**Relevance:** Resolves a specific local lower-coefficient improvement requested in Dahan Section 7; does not automatically improve the global exceptional-set exponent until downstream hypotheses are checked.
