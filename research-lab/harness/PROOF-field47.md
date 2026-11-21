# Finite-field root certificate for a forbidden five-hole shape

**Date:** 2026-11-03  
**Claim:** MOD5-FIELD47-20261103  
**Validity:** ANALYTICAL_PROOF (short proof of a previously established restricted conclusion) + EXACT_FINITE_CERTIFICATE (finite-field witness).  
**Literature:** PRIORITY_UNCHECKED.  
**Formal:** UNFORMALIZED.  

## General character-root obstruction (elementary lemma)

Let N be odd, and let A be an n-element multiset of residues modulo N whose 2^n **labelled** subset sums are all distinct. Let H be the set of missing residues, so |H|=N-2^n, and write H(X)=sum_{h in H}X^h in Z[X]/(X^N-1). Let K be a field of odd characteristic and r in K a nontrivial element of odd order dividing N. Then **H(r) is nonzero**.

**Proof.** The distinct-subset-sum hypothesis gives the identity

    product_{a in A}(1+X^a) = (1+X+...+X^(N-1)) - H(X)   mod (X^N-1).

Evaluation at r is well-defined because r^N=1. The geometric series vanishes because r!=1 and r^N=1. Hence product_a(1+r^a)=-H(r). Each factor is nonzero: if r^a=-1, the left side has odd order while -1 has order 2 in K. Therefore H(r)!=0. QED.

This is a general necessary condition, not a sufficiency criterion. The proof makes no assumption that generators a are units modulo N. It requires N odd and the field characteristic not 2.

## Explicit witness for radius-(1,4) holes

Let

    P(X)=X^8+X^5+X^4+X^3+1,
    p=1129,
    r=45 in F_p.

The integer p is prime: trial division by all primes up to sqrt(1129)<34 gives no divisor. Direct finite-field modular arithmetic gives

    45^47 = 1 (mod 1129), 45 !=1,
    45^3=805, 45^4=97, 45^5=978, 45^8=377 (mod 1129),
    P(45)=377+978+97+805+1=2258=2*1129 =0 (mod 1129).

Since 47 is prime, r has exact multiplicative order 47. This produces an explicit root of the hole polynomial at an odd-order character.

For any integer k>=0, n=9+23k satisfies 2^n+5 divisible by 47: 2^9+5=517=11*47 and 2^23=1 (mod 47). Thus for N=2^n+5, r^N=1. If a set with all labelled subset sums distinct left exactly H={c,c±1,c±4}, then

    H(r)=r^(c-4) P(r)=0,

contradicting the general character-root lemma. Therefore there is no such set for any k>=0 and any center c. QED.

The proof does not use the previously computed 46x46 determinant, the resultant formula, the condition that n be odd or any group-classification theorem.

## Independent finite-field check

The two bounded workers agree that the complete common-root set in F_1129 of F_47 and P is {45,276}. An independently implemented polynomial Euclidean algorithm returns

    gcd_{F_1129[X]}(F_47,P)=X^2+808X+1
                                 =(X-45)(X-276).

The reciprocal roots reflect the palindromic polynomial P. Control: gcd(F_11,P)=1 over F_1129, P(1)=5, and replacing P by P+1 breaks both displayed roots. These are validation controls; they do not prove novelty.

## Scope and impact

This replaces the large exact resultant certificate for the already established 2026-11-02 forbidden-hole progression with a short direct finite-field witness. It does **not** improve the range of n, classify all t=5 sets or solve the integer Erdős distinct-sums problem. The general character-root test is a reusable necessary condition for other hole patterns, but no new pattern has been resolved here. Literature priority is unverified.

Strongest conventional explanation: the finite-field witness was expected from the previously known factor 1129^2 in the resultant and 47|1128. A proof simplification is meaningful verification work, not necessarily original mathematics.

## Harness integration

This copy is a verification fixture for the 3 November visual/harness update. It does not extend the mathematical conclusion. The registry checks the arithmetic obligations only; ordinary proof and literature priority remain separate.
