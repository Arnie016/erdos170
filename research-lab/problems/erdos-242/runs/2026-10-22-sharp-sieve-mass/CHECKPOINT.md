# 2026-10-22 checkpoint: Erdős #242 auxiliary sieve mass

**Verified here:** ordinary proof that W(J)=9/(2π²)·log²J+O(log J), where W(J) is Dahan's coprime sum of 1/φ(4ua). Proof gives an exact leading constant, exceeding the author's requested 0.45 threshold. One standard-library worker completed precise checks. Status ANALYTICAL_PROOF / PRIORITY_UNCHECKED / UNFORMALIZED.

**What changed:** Earlier Lemma 4.22 lower coefficient was 3/(2π²). The analytical input is sharpened; no all-n Erdős–Straus result or downstream exponent is certified.

**Next single discriminating step:** substitute the new coefficient into Dahan Lemma 4.23, Theorem 4.28 and Corollary 4.31, audit all extra hypotheses and recompute the exceptional-set bound or show why the substitution cannot pass. Avoid further small-n scans and repeated c=3/c=7 first-denominator searches.
