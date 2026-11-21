# Five missing residues in the modular distinct-subset-sums problem

**Status:** ANALYTICAL_PROOF (ordinary proof), PRIORITY_UNCHECKED, UNFORMALIZED.
**Scope:** Every integer n >= 2, not just the previously certified n=6,7,8 instances.

## Theorem

Let N=2^n+5 for n>=2. Let A be an n-element subset of Z/NZ with all 2^n labelled subset sums different. Write S for the represented sums, M for the five missing residues, T for the sum of all members of A, and c=T/2 mod N (unique because N is odd).

1. If d divides N and k>=1 members of A vanish modulo d, then the number of missing points in *every* fiber mod d is congruent to N/d modulo 2^k. Consequently d*((N/d) mod 2^k)<=5, using the least nonnegative remainder.
2. If n is odd, every member of A is invertible modulo N. If n is even, at most one member of A is not invertible and that member necessarily has gcd(a,N)=3.
3. There exist nonzero radii r,s with five distinct points such that M={c,c+r,c-r,c+s,c-s}.
4. If n is even and t∈{0,1} counts the noninvertible marks, the missing points have fiber counts (1+2t,2-t,2-t) in residue classes (c,c+1,c-1) modulo 3. Equivalently, exactly t of the radii r and s are divisible by 3.

## Proof

**Fiber divisibility.** For each choice of subset among the n-k marks not divisible by d, its 2^k extensions using the k divisible marks share the same residue modulo d. Injectivity of all labelled subset sums makes these 2^k extensions distinct residues modulo N. Thus the represented subset-sum count in every fiber is divisible by 2^k. Each fiber contains m=N/d residues. Therefore its number of missing residues δ obeys δ≡m mod 2^k. Since m is odd, δ is at least the positive remainder m mod 2^k. Summing over d fibers gives d*(m mod 2^k)<=5.

**Nonunits.** If gcd(a,N)=g>1, use the fiber bound with d=g, k>=1. Then g<=5. Both g and N are odd, and 5 cannot divide N=2^n+5, so g=3. For odd n, N≡1 (mod 3); hence there are no nonunits. For even n, N/3=(2^n+5)/3≡3 (mod 4), since 2^n≡4 (mod 12). If two or more marks were divisible by 3, take d=3 and k>=2: each missing-fiber count is congruent to 3 modulo 4 and hence at least 3, so |M|>=9, contrary to |M|=5. Thus at most one nonunit exists, of gcd 3.

**Center.** Complementing a subset sends its sum x to T-x. Consequently both S and M are stable under x↦T-x. This map has exactly one fixed point c=T/2 because N is odd. The set S has even cardinality 2^n, so cannot contain the unique fixed point. Hence c∈M, and the other four missing residues form exactly two complementary pairs. This proves the shape in (3).

**Modulo-3 fingerprint.** Take even n, and t=0 or 1, the number of elements divisible by 3. Let P(x)=∏_{a∈A}(1+x^{a mod 3}) in Z[x]/(x^3−1), with coefficient R_i counting distinct subset sums in residue class i modulo 3. The identities 1+x≡−x² and 1+x²≡−x modulo 1+x+x² show

P(x)≡2^t(−1)^{n-t} x^{-T}  (mod 1+x+x²).

Together with P(1)=2^n, this gives, for h≡−T (mod 3):

- t=0: R_h=(2^n+2)/3 and R_{h±1}=(2^n−1)/3;
- t=1: R_h=(2^n−4)/3 and R_{h±1}=(2^n+2)/3.

Because all subset sums are distinct mod N, these are also the actual occupied counts in the three fibers of Z/NZ. Each fiber has (2^n+5)/3 positions, leaving respectively (1,2,2) or (3,1,1) missing positions. Since 2^{-1}≡−1 modulo3, h≡c modulo3. A pair {c+r,c-r} contributes two points to the c fiber exactly when r≡0 modulo3, otherwise one to each off-center fiber. The observed counts therefore force exactly t radii divisible by3. QED.

## Comparison and limitation

The project’s 2026-10-23 modulo-261 certificate already proved the local gcd-3 obstruction at n=8; the present argument extends it to every n and adds the exact five-hole residue fingerprint. Published t=5 perturbation templates and general parity methods are due to Cambie, Gao, Kim and Liu. Neither this lemma nor the finite checks prove that all admissible n-sets belong to the known three signed-unit template classes or are within one replacement of powers of two. Exact literature priority for this combined lemma has not been audited.