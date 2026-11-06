# Exact classification for seven modular subset-sum generators

**Claim:** MOD133-2026-10-17. **Status:** EXACT_FINITE_CERTIFICATE (local execution); repository CI reproduction pending. **Literature:** PRIORITY_UNCHECKED. **Lean:** UNFORMALIZED.

For N = 2^7+5 = 133, a 7-element subset A of the nonzero residues is admissible if all 128 labelled subset sums, including zero, are pairwise distinct. Two admissible sets are equivalent when one is obtained from the other by multiplying all entries by a unit modulo 133 and changing signs of any entries independently.

**Finite theorem.** There are exactly 20,736 admissible seven-element sets. Up to the stated equivalence, exactly three classes occur, represented by:

* A0 = {1,2,4,8,16,32,64}
* A1 = {1,2,4,8,16,32,65}
* A2 = {1,2,4,8,16,33,65}

Each class contains 54 sign-normalized sets. Every admissible set shares at least 6 elements with a signed, unit-scaled copy of the powers-of-two set. This is a finite N=133 result, **not** an all-n theorem or a solution of the classical integer Erdős distinct-sums question.

## Reduction

An admissible A cannot contain 0 or both x and -x, because distinct labelled subsets then have the same sum. Replacing one entry by its negative translates its subset-sum set and preserves injectivity. As N=133 is odd, the 132 nonzero residues form 66 opposite pairs. Every admissible set thus has exactly one sign-normalized image in the 7-subsets of {1,...,66}, and exactly 2^7 = 128 distinct sign lifts.

The Python generator uses a 133-bit set S of subset-sum residues. To add a candidate a, check S ∩ (S+a) is empty; then update S to S ∪ (S+a). All strictly increasing canonical choices are enumerated, with only collision-bearing prefixes discarded.

The C++ independent checker instead maintains D=S-S, the set of differences between older subset sums, with D initially {0}. A mark a can be added if and only if a is not in D. Then D becomes D ∪ (D+a) ∪ (D-a). This follows algebraically by expanding (S ∪ (S+a))-(S ∪ (S+a)). At accepted leaves the checker *separately* reconstructs all 128 subset sums in an ordinary ordered set. It computes all unit-multiple/sign-normalized orbits directly, rather than trusting a search score.

The complete depth-0 through depth-7 counts were identical in the two algorithms:

1, 60, 1830, 36656, 490780, 3322318, 2942381, 162.

The C++ checker partitions all 162 canonical admissible sets into 3 unit orbits, each of size 54. The maximum intersection with a signed-unit copy of powers of two is 7 for 54 sets and 6 for 108 sets. Multiplying by 128 sign lifts yields 20,736.

Each representative is superincreasing as an ordinary increasing integer sequence, and its total sum is respectively 127, 128 or 129, smaller than 133. Therefore every representative has 128 distinct ordinary subset sums and consequently distinct residues modulo 133. A0 shares 7 elements with the standard powers-of-two set; A1 is obtained by replacing 64 with 65; A2 is obtained by replacing 66 with 65 in the scaled powers-of-two set {1,2,4,8,16,33,66}, corresponding to multiplier 33. This supplies the explicit one-replacement witness for every class, by invariance under units and signs.

The two tested source files and per-depth counts form an exact finite certificate; neither program is a Lean proof or an unbounded proof. The canonical input domain has C(66,7)=778,789,440 combinations before justified prefix pruning. A negative-control set {1,2,3,4,5,6,7} gives only 29 distinct subset sums. No GPU or paid service was used.

**Historical prerequisite:** n=6, N=69 has an independently checked census of 66 canonical sets in this repository. We did not rerun it today.

**Literature boundary:** Cambie, Gao, Kim and Liu, *The Erdős distinct subset sums problem in a modular setting*, arXiv:2308.03748 (2023, Acta Arithmetica 2025), Section 7 and Proposition A.1 already propose perturbations at t=5. The three templates above instantiate those published constructions. This new computation validates completeness only at n=7. New-priority literature was not exhaustively checked.

**Next falsifier:** an admissible n=8 set modulo 261 outside the corresponding three signed-unit classes would refute the proposed all-n classification. Before another full enumeration, investigate a universal restriction on the five missing subset-sum residues.
