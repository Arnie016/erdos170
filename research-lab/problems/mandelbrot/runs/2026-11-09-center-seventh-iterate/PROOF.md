# MAND-CENTER7-2026-11-09-A | exact seventh-iterate classification

**Validity:** ANALYTICAL_PROOF + EXACT_FINITE_CERTIFICATE. **Literature:** known method, specific instance PRIORITY_UNCHECKED. **Formalization:** UNFORMALIZED. **No new area certificate.**

Define z₀(c)=0 and zₙ₊₁(c)=zₙ(c)²+c. The finite domain is the 22 centers from the 8 November exact result (`dd482e9e27cba1406fa83cfd981ece5fb7cc2c32`) with |z₆(c)|²<4. They come from nine parent grid masks from 7 November (`c6bfd7363fabe6ba52dd7c28571eeba380d33c42`). For a zero-bit child (i,j,a,b), its center is c=(p+iq)/8000 with p=-6000+8i+2a+1 and q=4000+8j+2b+1, where a,b range over 0,1,2,3.

**Theorem.** For **every one of these 22 rational parameters**, |z₆(c)|²<4 and |z₇(c)|²>4, strictly. There are no equality cases. Hence all 22 points lie outside the Mandelbrot set under the standard quadratic escape-radius criterion; combined with the previous 20 already escaping centers, all 42 formerly uncertified *centers* are exterior by iteration seven. **Nothing is proved about the entire child squares**, and previously certified exterior area is unchanged.

Worker 1 calculates exact Gaussian-rational coordinates through seven recurrences. Worker 2 uses independent homogeneous Gaussian integers z=(A+iB)/D, with updates A'=8000(A²−B²)+pD², B'=16000AB+qD² and D'=8000D², reducing the common gcd at every step. Both reconstruct the sixth-step exclusion condition and agree on all 22 reduced seventh-step squared-norm fractions. These exact fractions have approximate range 9.980655668 to 10.282286877, well above the threshold 4. Decimal values are **not** the acceptance test.

**Frozen falsifier:** any non-22 domain, a failed or timed-out worker, different rational norms, a z₆ point failing the precondition or a seventh norm ≤4. Controls include c=0 and the reference parameter c₀=-3/4+i/2. Both original workers exited 0 with respective wall times 0.76044 s and 0.64851 s under 35-second limits.

**Implication:** proves the 6-iterate obstruction was a horizon limitation at all 22 centers, not evidence of Mandelbrot membership. Moving from points to complete cells still requires uniform seventh-step enclosures. Next research should rotate toward sparse/dense combinatorics instead of more center-only repetitions.
