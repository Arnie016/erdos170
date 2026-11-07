# MAND-SUBGRID-2026-11-07-A: exact fourfold refinement

**Status:** ANALYTICAL_PROOF + EXACT_FINITE_CERTIFICATE. **Method:** KNOWN. **Literature priority:** UNCHECKED. **Lean:** UNFORMALIZED.

Let M be the set of complex parameters for which z₀=0, zₙ₊₁=zₙ²+c is bounded. Put c₀=−3/4+i/2. The 6-Nov study at commit 762c3437602572eaca346d2dae70d37c1586058d certified 207 of 216 closed annular squares with side 1/1000 and left nine uncertified. Their exact indices are recorded in CERTIFIED_SUBCELLS.json.

For a parent (i,j), define its 16 closed children C(i,j,a,b)=c₀+([4i+a,4i+a+1]+i[4j+b,4j+b+1])/4000, where a,b=0,1,2,3.

**Theorem.** Each child with bit (4a+b) set in the accompanying parent hex mask is entirely outside M. There are exactly 102 such children in the frozen domain. Two whole parent squares are now certified. The other 42 children remain uncertified.

**Proof method 1.** Build the Gaussian-rational polynomial P₆(t)=z₆(c₀+t), degree 32, using P₀(t)=0 and Pₙ₊₁(t)=Pₙ(t)²+c₀+t. Rational complex interval Horner evaluation encloses its values over the entire child rectangle by interval endpoint extrema. For resulting intervals R,I for real and imaginary parts, put d([a,b])=0 if a≤0≤b, else min(|a|,|b|). Accept a child only when d(R)²+d(I)²>4 exactly.

**Proof method 2.** Independently construct the integer polynomial Z₆(c) from Z₀=0, Zₙ₊₁=Zₙ²+c. Binomially translate at the child center γ to obtain Z₆(γ+u)=Σbₖuᵏ over exact Gaussian rationals. Its half-diagonal is √2/8000<q=1/5600. Bound E=Σₖ₌₁³²(|Re bₖ|+|Im bₖ|)qᵏ and accept only if |b₀|²>(2+E)².

Both tests imply |z₆(c)|>2 for every c in the accepted child. Here |c|<2, and the reverse triangle inequality |zₙ₊₁|≥|zₙ|²−|c| implies an orbit with |z₆|>2 diverges. The first and second methods locally accepted 105 and 102 children respectively. The 102-cell **intersection** is the claim; three worker-1-only children are not counted.

Each new accepted child has interior area 1/16,000,000. Its interior is disjoint from the old 207 accepted squares and earlier radius-1/250 disk. The total certified exterior area is therefore at least π/62,500+207/1,000,000+51/8,000,000, approximately 5.245 times the 5-Nov disk alone.

**Limits.** The other 42 children are not certified, not known to belong to M. The study establishes no global boundary theorem, no Hausdorff dimension or novelty. Local worker passes are not GitHub dedicated CI or Lean formalization.
