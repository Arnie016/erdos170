# Correcting the alleged non-template five-hole shape

**Claim MOD5-SHAPE-CORR-20261027.** **CORRECTION**, construction **KNOWN**, **UNFORMALIZED**. This repairs a planned candidate selection; it is not a new solution, global classification or priority claim.

## Claim contract

For each integer n≥3, N=2^n+5, consider five missing subset-sum residues of the form {c,c±1,c±3}. The proposed n=9 test treated these centered radii as outside the recorded perturbation templates. A valid family for all n together with a signed-unit equivalence to a known template falsifies that description. Stop at such a construction; do not repeatedly test this family as an allegedly new obstruction.

## Ordinary proof

Take A_n={2,3,4,8,16,...,2^(n−1)}. The eight subset sums of {2,3,4} are exactly U={0,2,3,4,5,6,7,9}, one representative of each residue class mod 8. The remaining powers have subset sums 8k, 0≤k<2^(n−3). Hence all sums 8k+u are different as ordinary integers: equal sums would have equal u modulo 8 and therefore equal u and k.

Writing M=2^n, these sums fill {0,...,M−1} except 1 and additionally contain M+1. The largest is M+1<N. Thus A_n has all M labeled subset sums distinct modulo N; its missing residues are {1,M,M+2,M+3,M+4}. Their center is c=−2 (mod N), and the missing set is precisely {c,c±1,c±3}.

This is not a new family. Let D_n={1,2,4,...,2^(n−2),2^(n−1)+1}. Multiplication by 2, a unit mod the odd integer N, changes its last entry to M+2≡−3. Changing this entry's sign to +3 gives exactly A_n. Sign changes merely translate the complete subset-sum set, and unit dilation preserves distinctness. Consequently the ±1,±3 shape lies in an established signed-unit perturbation orbit.

## Pre-screen before a further search

Under independent sign changes, the missing-hole set translates, so its two centered radii do not change. Under multiplication of generators by a unit, both radii are multiplied by that unit. With unit radii, the ratio is preserved up to signs and inversion.

At n=9, N=517, the three previously recorded template representative shapes have radii {1,2}, {1,3} and {1,130}. The proposed next shape {1,4} is not unit-equivalent to these: 4 and its signed reciprocal generate {4,513,388,129}, which contains none of 2,3,130. This is only a filter against the **recorded** representatives; it is not a literature-complete nonexistence argument.

**Scope:** no n=9 nonexistence theorem, universal t=5 classification or integer Erdős distinct-sums claim. This correction changes the next test, not the global mathematical baseline.
