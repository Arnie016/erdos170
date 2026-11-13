# MAND-CENTER-2026-11-08-A | sixth-iterate method obstruction

**Validity:** ANALYTICAL_PROOF + EXACT_FINITE_CERTIFICATE (42 exact rational point evaluations). **Method:** known; specific priority unchecked. **Formalization:** UNFORMALIZED. **No new area certificate.**

Let c₀=−3/4+i/2 and z₀(c)=0, zₙ₊₁(c)=zₙ(c)²+c. Use the nine parent indices and four-by-four child masks in the 7-Nov certificate at commit `c6bfd7363fabe6ba52dd7c28571eeba380d33c42`, `CERTIFIED_SUBCELLS.json`. Define the closed children
`C(i,j,a,b)=c₀+([4i+a,4i+a+1]+√(-1)[4j+b,4j+b+1])/4000`, where a,b range from 0 through 3. Investigate precisely the **42 zero-bit** children, with bit index 4a+b. Their centers are `γ=(p+√(-1)q)/8000`, where `p=−6000+8i+2a+1` and `q=4000+8j+2b+1`.

**Theorem.** Exactly **20** of these centers satisfy |z₆(γ)|²>4; **22** satisfy |z₆(γ)|²<4; none satisfies equality. Therefore 20 individual centers lie outside M. For each of the other 22 closed child squares, the full-square condition `∀c∈C, |z₆(c)|>2` is **false** because γ∈C is a counterexample.

This is a structural obstruction to one *sixth-iterate* proof method, not a claim of Mandelbrot membership: those center orbits may escape after a later iterate. A pointwise calculation does not certify a child square, so no previously published area ratio changes.

**Exact proof and cross-check.** First checker iterates the two Gaussian-rational coordinates six times and tests the exact rational squared norm against 4. Independently, put γ=(p+iq)/8000 and write z=(A+iB)/D. The second checker uses integers:
```
D' = 8000 D²
A' = 8000(A² − B²) + pD²
B' = 16000AB + qD².
```
Starting (A,B,D)=(0,0,1), after six steps |z₆|²=(A²+B²)/D² exactly. The checkers agree on **every one of 42 exact fractions** with separate c=0 and c₀ controls. For example, child (5,-8,0,0) has |z₆|²≈3.9965607457050<4; child (3,-9,2,0) has |z₆|²≈4.0136779503144>4. Decimal displays are not used for certification.

The nonescaping-by-six center counts, by parent, are (5,-8):10, (6,-7):6 and (7,-6):6. All other listed parents have zero such centers. Parent masks and selected cell results are fixed in the local exact output receipts; the illustration is a labeled explanation, not an independent proof.

**Strongest limitation:** 22 center nonescapes do not determine eventual behavior, while the remaining 20 exterior centers do not prove exterior for their whole squares.

**Next discriminating test:** compute the **seventh** exact critical iterate at precisely the 22 centers with sixth-iterate norm ≤2. If those escape by seven, the obstruction is due to the frozen horizon; if not, record the remaining delayed cases.

Provenance: Prior claim MAND-SUBGRID-2026-11-07-A (PR #8), standard quadratic escape-radius lemma. No global result, Hausdorff-dimension inference, literature novelty or Lean proof is claimed.
