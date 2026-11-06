# Next test | 2026-11-01

MOD5-R4-230-20261101: ANALYTICAL_PROOF + EXACT_FINITE_CERTIFICATE, PRIORITY_UNCHECKED, UNFORMALIZED.

At N=517, the exact resultant Res(F_517,X^8+X^5+X^4+X^3+1)=4493599431468313730139668192986121524839260273570500546030600490466361. Two integer implementations independently produced the same answer, and both positive and zero-resultant controls passed. It is not 1. Since F_517 divides F_N whenever 517 divides N, and unit generators force the five-hole norm to equal 1, the centered radius-(1,4) holes are impossible for every n=9+230k with N=2^n+5. The n=9 exclusion itself was previously known in this project from the 30 October graph certificate.

**One next discriminating test:** compute the separate cyclotomic norms for divisors d=11,47,517. A nonunit factor for a smaller d could exclude additional congruence classes of n. Do not assume such a factor without exact proof. Do not repeat the old n=9 graph search.

No priority or full Erdős solution claim, no Lean verification, no outreach authorizations, no schedule change.