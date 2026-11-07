# MAND-EXTERIOR-2026-11-05-A | Fourfold larger certified exterior disk

**Scope:** Exact parameter disk exclusion for quadratic Mandelbrot critical iteration. **Validity:** ANALYTICAL_PROOF + EXACT_FINITE_CERTIFICATE. **Literature:** KNOWN METHOD, exact example PRIORITY_UNCHECKED. **Formalization:** UNFORMALIZED.

Let M be the set of complex parameters c for which the critical orbit z_0=0 and z_(j+1)=z_j²+c remains bounded. Fix c0 = -3/4 + i/2 and r=1/250.

**Theorem.** Every c with |c-c0| <= r lies outside M, because |z_6(c)|>2. The previously certified closed disk at this center had radius 1/1000. The new radius is 4 times as large, so the certified disk area is 16 times as large.

**Proof.** Set t=c-c0 and write z_6(c0+t)=a_0+a_1 t+...+a_32 t^32 with a_k in Q(i). Each coefficient is uniquely determined by P_0(t)=0 and P_(j+1)(t)=P_j(t)^2+c0+t. For |t|<=r,

    |z_6(c0+t)-a_0| <= sum_(k=1)^32 |a_k| r^k
                           <= S(r):=sum_(k=1)^32 (|Re a_k|+|Im a_k|) r^k.

S(r) is a nonnegative rational number. The exact rational checker worker2.py constructs the integer polynomial z_6(c), translates it by the binomial theorem and checks, with no floating-point arithmetic:

    |a_0|^2 > (2+S(1/250))^2.

Therefore |z_6(c)| >= |a_0|-S(r)>2 throughout the entire closed disk. Also |c|<=3/4+1/2+1/250<2. By |z_(j+1)| >= |z_j|^2-|c| > |z_j|^2-2, once the orbit exceeds 2 it diverges. QED.

The exact center iterate is

    a_0 = -39742475709585026047/18446744073709551616
          + (55868220783672119/288230376151711744) i.

The independent exact checker found S(1/250) approximately 0.1080677930, center norm squared approximately 4.6791994046 and positive squared margin approximately 0.2352495845. These decimals only aid interpretation; the strict inequality is evaluated with exact fractions. The other checker, worker1.py, generates Taylor coefficients directly in t and obtains outward rational upper bounds for their complex magnitudes using integer square roots. Its stronger error bound at r=1/250 is approximately 0.0769888348. Both algorithms reconstructed identical 33 rational complex coefficient pairs.

**Frozen domain:** radii 1/1000, 1/750, 1/500, 1/400, 1/250 and 1/100. Both exact methods certify the first five. Neither method certifies 1/100; this is not proof of Mandelbrot membership at that larger radius. This result is a local increase of certified excluded area, not a new global Mandelbrot theorem, Hausdorff-dimension result or proof of novelty.

