# A published clique witness with three incorrect printed Gram entries

**16 October 2026 | Erdős #117 | SOURCE-TABLE CORRECTION | No global theorem claim**

The 13 explicitly printed vectors in Guillaume Lecomte's public preprint [arXiv:2608.20507v1](https://arxiv.org/pdf/2608.20507), Section 4, Lemma 4.4, have **all 78 pairwise symplectic products nonzero** over F3. This is the property needed for the cited finite nonorthogonality witness. However **three of the paper's 78 printed Gram entries disagree with those exact vectors**. The original PDF page 5 was visually checked against the HTML, so this is not an HTML-only transcription artifact.

Read the [explicit proof and limitation](proofs/GRAM_CORRECTION.md), the [attributed public finite source data](data/lecomte_13_vectors.json), the [78-pair diagnostic](data/DIAGNOSTIC.json) and the [two checkers](src/). The initial *strict* checker failed (exit 1, as it should when comparing with the printed Gram table). The final independent wedge-minor diagnostic completed (exit 0) and identified the three mismatches. **Do not mistake diagnostic completion for strict acceptance.** Both sequential processes had 35-second timeout caps, and no third worker was used. No Lean execution was performed. The original full paper was not independently proved.

Reproduce the diagnostic from this directory using `python3 src/diagnose_source_mismatch.py`. Running `python3 src/check_lecomte_13.py` should FAIL on the original printed-table acceptance condition. Source vectors and printed entries are attributed to the public article under its indicated CC BY 4.0 terms; our checkers and explanations are project-authored. No private correspondence or unpublished manuscript was used.

**Research impact:** no change to the best all-group h(n) bound, no claim of Erdős #117 being solved by this audit. Future #117 work should prioritize genuinely undecided finite-threshold or uniform-constant questions rather than a known-correct 13-clique existence witness.
