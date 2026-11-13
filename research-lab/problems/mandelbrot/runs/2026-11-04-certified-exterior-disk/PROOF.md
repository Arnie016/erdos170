# MAND-EXTERIOR-2026-11-04-A | Exactly certified exterior disk

**Scope:** One closed region of the Mandelbrot parameter plane. **Validity:** ANALYTICAL_PROOF + EXACT_FINITE_CERTIFICATE. **Literature:** KNOWN METHOD, priority for this illustration unchecked. **Formalization:** UNFORMALIZED.

Define
\[
M=\{c\in\mathbb C: z_0(c)=0,\ z_{j+1}(c)=z_j(c)^2+c\ \text{is bounded}\}.
\]

Fix \(c_0=-3/4+i/2\), \(\rho=1/1000\) and \(D=\{c:|c-c_0|\le \rho\}\).

**Claim.** Every \(c\in D\) satisfies \(|z_6(c)|>2\), hence \(D\cap M=\varnothing\).

**Proof.** Let \(w_j=z_j(c_0)\) and \(d_j=z_j(c)-w_j\). Since the center has rational coordinates, so does every \(w_j\). Set \(U_j=|\Re w_j|+|\Im w_j|\), a rational upper bound on \(|w_j|\). Define
\[
E_0=0,\qquad E_{j+1}=(2U_j+E_j)E_j+\rho.
\]
The exact perturbation identity is
\[
d_{j+1}=2w_jd_j+d_j^2+(c-c_0).
\]
By induction and the triangle inequality, \(|d_j|\le E_j\) for every \(c\in D\).

The exact rational calculation in [CERTIFICATE.json](CERTIFICATE.json), reproducible with [worker1.py](worker1.py), establishes
\[
|w_6|^2>(2+E_6)^2.
\]
It follows that \(|z_6(c)|\ge |w_6|-E_6>2\) uniformly over the entire parameter disk.

Also \(|c|\le |c_0|+\rho\le 5/4+1/1000<2\). Once any orbit with \(|c|<2\) reaches magnitude \(R>2\), the reverse triangle inequality yields
\[
|z_{j+1}|\ge |z_j|^2-|c|>|z_j|^2-2>|z_j|.
\]
For each further step the growth factor is at least \(R-2/R>1\), where \(R>2\) is the first escaped magnitude. Thus the orbit diverges. QED.

The exact center sixth iterate is
\[
w_6=-\frac{39742475709585026047}{18446744073709551616}
+\frac{55868220783672119}{288230376151711744}\,i.
\]
The exact error \(E_6\) appears in the JSON. Approximate decimals, the visualization and finite nonescape are **not** used to prove membership or global properties.

**Independent challenge and limitation:** The first rational worker exited 0 after all tests. A second checker used Gaussian-integer numerator/denominator recurrences, reconstructed every center point and error term and rejected a deliberately corrupted certificate. Its mathematical assertions completed, but its final log print referenced a local variable incorrectly and the process exited 1. Therefore no clean independent second-worker success or new-study GitHub CI pass is claimed. No third experimental run/retry occurred. See [RUN_REPORT.md](RUN_REPORT.md).

**Prior art:** The escape criterion is classical and does not make this certificate a new global theorem. See [SOURCES.md](SOURCES.md). This diagram is one illustrative certified open-area exclusion, not a certified full Mandelbrot boundary, general membership algorithm or new dimension calculation.
