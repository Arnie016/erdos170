# Resultant obstruction on an infinite progression of modular subset-sum problems

**Claim MOD5-R4-230-20261101.** ANALYTICAL_PROOF + EXACT_FINITE_CERTIFICATE; literature PRIORITY_UNCHECKED; formalization UNFORMALIZED.

**Theorem.** For every integer k >= 0, let n=9+230k and N=2^n+5. No n-element set A modulo N has all 2^n labelled subset sums distinct with the five unrepresented residues of the form {c,c±1,c±4}, for any c.

**Proof.** Since 2^9+5=517=11*47, 2^10=1 (mod 11), and 2^23=1 (mod 47), we have 2^230=1 (mod 517). Thus 517 divides N for every n=9+230k. In particular n and N are odd. For odd n, N is coprime to 3 and 5, so every nontrivial divisor of N is at least 7.

Suppose A exists, and S is its set of distinct subset sums. Fix a in A, and write B for the sums formed without a. Distinctness gives S=B disjoint-union (B+a). Translation by a partitions Z/NZ into gcd(a,N) cycles, each of odd length. On a cycle with no missing residue, the binary indicator of B would have to alternate 0,1,0,1 around an odd cycle, impossible. Each cycle must contain one of the five missing residues. Hence gcd(a,N)<=5, forcing gcd(a,N)=1. All generators are units.

Let F_N(X)=1+X+...+X^(N-1) and H(X) the sum of X^h over all five missing residues h. Evaluating the subset-sum generating identity at every nontrivial N-th root of unity zeta gives

    product_(a in A) (1+zeta^a) = -H(zeta).

Taking the product over all zeta != 1 yields Res(F_N,H)=1: multiplication by a unit permutes the nontrivial roots, product_(zeta!=1)(1+zeta)=1 when N is odd, and there are N-1 even many minus signs.

For holes {c,c±1,c±4}, put P(X)=X^8+X^5+X^4+X^3+1. At any N-th root zeta, H(zeta)=zeta^(c-4)P(zeta). The product of the nontrivial N-th roots is 1, so Res(F_N,H)=Res(F_N,P).

The exact certificate below establishes

    R=Res(F_517,P)
     =4493599431468313730139668192986121524839260273570500546030600490466361 > 1.

Since 517 divides N, F_N=F_517*(1+X^517+...+X^(N-517)). Multiplicativity of the resultant gives Res(F_N,P)=R*Res(F_N/F_517,P), an integer divisible by R>1. It cannot equal 1. Contradiction. QED.

**Exact certificate.** Let C denote the integral multiplication-by-X matrix for Z[X]/(P). Because P is monic of even degree,

    Res(F_517,P) = det(C^517-I)/P(1).

Two separately implemented exact-integer programs computed the same determinant

    22467997157341568650698340964930607624196301367852502730153002452331805

and its quotient by P(1)=5. Worker 1 uses 8x8 companion-matrix exponentiation; worker 2 computes X^517-1 modulo P, then the Sylvester resultant. Both use fraction-free integer determinant elimination. The known realizable radius-(1,3) polynomial gives resultant 1, and Phi_11=1+...+X^10 gives resultant 0. These controls passed. Local execution receipts, code hashes, and the complete outputs are available in this run's local evidence package; the uploaded code is included for a fresh reproduction. There is no Lean proof.

**Scope and priority.** The finite case n=9 had already been excluded by a different bipartite compatibility-graph certificate on 30 October. This run extends the conclusion to every n=9+230k without rerunning that census. Nothing is claimed about other hole shapes, other congruence classes, or the entire distinct-sums problem. Related constructions are treated in Cambie–Gao–Kim–Liu, Acta Arithmetica 217 (2025); whether this specific restricted observation appears elsewhere is PRIORITY_UNCHECKED. The all-odd-modulus norm identity is not claimed original.