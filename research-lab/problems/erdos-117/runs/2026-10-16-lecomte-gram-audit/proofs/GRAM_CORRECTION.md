# Exact finite check: 13 public F3 vectors and their printed Gram triangle

**Scope:** CORRECTION to a numerical table, not a counterexample to a published existence lemma or an all-group theorem. Literature priority **PRIORITY_UNCHECKED**.

For vectors in F3^6 use the exact alternating form

\[ B(x,y) = x_1 y_2 - x_2 y_1 + x_3 y_4 - x_4 y_3 + x_5 y_6 - x_6 y_5 \pmod 3. \]

In [Guillaume Lecomte, *Sharp Asymptotics for Abelian Covers of Groups with Bounded Noncommutativity*, arXiv:2608.20507v1, Section 4, Lemma 4.4](https://arxiv.org/pdf/2608.20507), thirteen public vectors and their claimed strict upper-triangular Gram matrix are displayed. We visually checked the original v1 PDF page 5 and the arXiv HTML, which agree. From the displayed vectors, exactly **75 of the 78** printed Gram values match. The other three are:

| 1-based pair | Direct value | Printed value | Three 2×2 minors mod 3 |
|---|---:|---:|---|
| (5,9) | **1** | 2 | 2 + 0 + 2 = 4 ≡ 1 |
| (9,12) | **2** | 1 | 2 + 1 + 2 = 5 ≡ 2 |
| (10,13) | **2** | 1 | 0 + 0 + 2 = 2 ≡ 2 |

For (5,9), the corresponding vectors are (1,2,1,0,2,0) and (1,1,1,0,0,1). For (9,12), they are (1,1,1,0,0,1) and (1,0,1,1,1,1). For (10,13), they are (1,2,0,0,2,1) and (1,2,0,1,1,0).

**Positive finite result:** all 78 actual values are 1 or 2 modulo 3; the 13-point nonorthogonality witness survives. **Negative finite result:** the printed Gram triangle is not exactly the Gram triangle of the displayed vectors. This is not a refutation of Lemma 4.4's existence claim. No derivation here improves or invalidates the all-group function h(n)=sup{a(G): ω(G)≤n}. No claim of novelty of the correction is made.

**Predeclared acceptance and failure:** the strict initial worker required 78/78 nonzero **and** 78/78 printed matches, plus rank-6 and negative-control checks. It exited 1 at `AssertionError: source verification FAIL` before rank and control, so those later checks are **unexecuted**. A separately frozen diagnostic then recomputed all 78 via 2×2 minors, producing 75 printed matches, 3 mismatches and no zeros; its successful process exit does not repair the original failed acceptance. Two workers only, each under a hard 35-second timeout (1.09 sec and 0.76 sec observed). Source coordinates and table are in `../data/lecomte_13_vectors.json`; report in `../data/DIAGNOSTIC.json`. Exact modular expressions above are independently checkable without trusting code.

The published v1 PDF and HTML were inspected, not the source TeX; an author erratum or later revised preprint may supersede the correction. This source audit is bounded and does not constitute a complete independent verification of Lecomte's main theorem. The copied short finite data are attributed under the paper's stated CC BY 4.0 terms.
