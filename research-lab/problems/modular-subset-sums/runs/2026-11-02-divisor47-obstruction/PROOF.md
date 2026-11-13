# The divisor-47 cyclotomic norm excludes radius-(1,4) holes on a denser infinite progression

**Validity:** ANALYTICAL_PROOF + EXACT_FINITE_CERTIFICATE  
**Literature:** PRIORITY_UNCHECKED  
**Formalization:** UNFORMALIZED  
**Claim ID:** MOD5-DIV47-20261102

## Theorem

For every integer k >= 0, put n=9+23k and N=2^n+5. No n-element subset A of Z/NZ can have all 2^n labelled subset sums distinct and leave exactly the five residues {c,c-4,c-1,c+1,c+4} unrepresented, for any center c.

The 2026-11-01 result handled only n=9+230k. The present result includes even n and improves progression density by a factor of ten. It does not settle other hole configurations or the unrestricted distinct-sums problem.

## Proof

N is odd. Let S be the 2^n represented residues and H the five missing residues. With F_N(X)=1+X+...+X^(N-1) and H(X)=sum_{h in H} X^h, the subset-sum identity in Z[X]/(X^N-1) is

prod_{a in A}(1+X^a) = F_N(X)-H(X).

For any Nth root of unity zeta != 1, F_N(zeta)=0, so prod_a(1+zeta^a)=-H(zeta). Set g=gcd(a,N). The map zeta -> zeta^a takes all Nth roots to all (N/g)th roots, each g times. Since N/g is odd, prod_{eta^(N/g)=1}(1+eta)=2. Removing the zeta=1 factor gives

prod_{zeta^N=1,zeta!=1}(1+zeta^a)=2^(g-1).

Multiply over all a. The N-1 minus signs cancel, proving the general necessary identity

Res(F_N,H)=2^E, E=sum_{a in A}(gcd(a,N)-1).

No assumption that the generators are units is needed. The right side is a pure power of two even when n is even.

For the prescribed holes, put P(X)=X^8+X^5+X^4+X^3+1. On Nth roots, H(X)=X^(c-4) P(X). The product of all nontrivial Nth roots equals 1 (F_N is monic of even degree and constant 1); hence Res(F_N,P)=Res(F_N,H)=2^E.

Now 2^9+5=517=11*47 and 2^23=1 (mod 47). Thus 47 divides N whenever n=9+23k. Consequently F_47 divides F_N in Z[X] and, by multiplicativity, Res(F_47,P) divides Res(F_N,P).

Two independent exact-integer programs give

Res(F_47,P)=1,274,641=1129^2.

This is an odd integer greater than one. It cannot divide any power of two. Contradiction.

## Independent exact certificate

P is monic of degree 8 with P(1)=5. For its 8x8 integral companion matrix C, Res(F_d,P)=det(C^d-I)/5. The computed exact values are Res(F_11,P)=1 (determinant 5), Res(F_47,P)=1,274,641 (determinant 6,373,205). One checker uses powers of C with Bareiss exact elimination. The other builds the integer Sylvester matrix of F_d and P directly. Both agree exactly.

Controls: Res(F_11,X^2+1)=Res(F_47,X^2+1)=1; Res(F_11,Phi_11)=0. Both checks exited zero under independent 35-second hard timeouts. See RUN_REPORT.md and the raw JSON files. No Lean formalization or new-study GitHub CI reproduction is asserted.

## Scope, priority and next test

The known algebraic norm identity is not claimed new. Priority of this particular forbidden-hole progression is unchecked. The d=11 norm is 1 and yields no obstruction. This result improves a restricted previous construction, not any general Erdős headline theorem.

**Next test:** seek an explicit small finite-field common-root certificate at the odd factor 1129, then check relevance to a predeclared neighboring hole shape without another broad census.
