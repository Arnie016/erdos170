# Unique inverse from the holes of a cyclic two-translate cover

**ID MOD5-INV-20261026.** ANALYTICAL_PROOF (ordinary mathematics), PRIORITY_UNCHECKED, UNFORMALIZED.

## Exact statement

Let N≥3 be odd, a a nonzero residue mod N, H⊆Z/NZ, and S=(Z/NZ)\H. Set g=gcd(a,N), L=N/g. The +a map partitions the residues into g cyclic orbits, each of odd length L.

There exists B⊆Z/NZ satisfying **S=B ⊔ (B+a)** (disjoint union) if and only if **every +a orbit meets H, and all cyclic step lengths between successive holes in that orbit are odd**. Such B is unique and has cardinality (N−|H|)/2.

## Proof

Enumerate one +a orbit as x_i=x_0+ia, i modulo L. Write f_i=1_{S}(x_i) and b_i=1_{B}(x_i). Disjointness and union are exactly the integer equations f_i=b_i+b_{i−1}. At every hole f_i=0, forcing b_i=b_{i−1}=0. A hole-free orbit would force b_i=1−b_{i−1} around an odd cycle and hence b_i=1/2, contradicting binary values.

Between successive holes d steps apart, f=1 at all intermediate positions, so b must alternate 1,0,1,... starting immediately after the first hole. The position immediately before the next hole must have b=0, exactly when d is odd. This proves necessity.

Conversely, when every orbit has holes separated by odd step lengths, assign b=0 at every hole and alternately 1,0 at the intermediate positions. All f_i=b_i+b_{i−1} equations hold and b is binary. Because one hole per orbit fixes two adjacent zeros, recurrence determines every remaining value uniquely. Finally the disjoint union shows |S|=2|B|. QED.

Algebraically, if T_a f(x)=f(x−a), then T_a^L=I and 2(1+T_a)^(−1)=∑_{j=0}^{L−1}(−1)^j T_a^j. Thus S and a uniquely determine the would-be half-set B even before testing its binary character.

## Consequence for the t=5 modular distinct-subset-sum problem

Let n≥2, N=2^n+5, A an n-set with all 2^n subset sums distinct, and H the five missing residues. For any genuine a∈A, B=S(A\{a}) is exactly the **unique** reconstructed set above, and must contain zero. This is a necessary all-n generator filter that uses only H and a. It does not prove B comes from n−1 other generators, nor that a passing candidate belongs to A.

Example of the necessary/sufficient distinction: A={1,2,4,8,16,32} mod 69 has H={64,65,66,67,68}. The inverse-hole test accepts a=10 with 0∈B, though 10∉A. Hence the single-generator test is not a complete classification.

## Limits and prior work

The 2026-10-24 project result already established the five missing residues' centered symmetry and gcd/fiber restrictions for all n, so those facts are prerequisites, not today's theorem. Cambie–Gao–Kim–Liu (arXiv:2308.03748; Acta Arith. 217, 2025), Lemmas 2.1–2.2 and Section 5, develop related parity alternation for t=1 and 3. No novelty or all-n dyadic-classification claim is made; this is an inspectable inverse criterion and a possible reduction for subsequent structural work. Two different finite checks tested four known witnesses, but they do not replace the ordinary proof.
