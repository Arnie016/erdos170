# First denominator of an infinite Erdős–Straus family

**P242-FIRST-20261020: ANALYTICAL_PROOF; PRIORITY_UNCHECKED; UNFORMALIZED.**

Define D(n) as the smallest positive integer x for which there are strictly ordered positive integers x<y<z with 4/n=1/x+1/y+1/z.

**Theorem.** If t is a positive integer divisible by 7 and every prime factor of t is congruent to 1 modulo 3, then for n=t²,

\[
D(n)=\frac{n+7}{4}.
\]

This includes t=7^j for every positive integer j. This is not a proof of the global conjecture.

## Proof

All prime divisors of t are 1 modulo 3. Thus t is odd, t≡1 (mod 6) and n=t²≡1 (mod 24). Positivity of the last two reciprocal terms forces x>n/4, so the smallest integer candidate is x0=(n+3)/4.

Set k=(t−1)/2. Then k≡0 (mod 3) and x0=k²+k+1. For every prime q dividing x0, q≠3 because x0≡1 (mod 3). The factorization (k−1)(k²+k+1)=k³−1 implies k³≡1 (mod q). If k≡1 (mod q), then q divides 3, a contradiction. Hence k has multiplicative order exactly 3 modulo q, so 3 divides q−1. Every prime divisor of x0 is therefore 1 modulo 3, as is every prime divisor of n.

Put m=n*x0. Suppose there are positive y,z with 4/n=1/x0+1/y+1/z. Cancelling 1/x0 gives 3/m=1/y+1/z, equivalent to

\[
(3y-m)(3z-m)=m^2.
\]

Each factor is strictly positive, because each reciprocal is less than 3/m. Both factors are congruent to −m≡2 (mod 3). But every positive divisor of m² is 1 modulo 3 because all prime factors of m are 1 modulo 3. This contradiction rules out x0.

Take x1=x0+1=(n+7)/4. Since 7 divides n, Q=n*x1/7 is a positive integer. Directly,

\[
\frac4n-\frac1{x_1}=\frac7{nx_1}=\frac1Q
=\frac1{Q+1}+\frac1{Q(Q+1)}.
\]

Since t≥7, n≥49 and Q=(n/7)x1≥7x1, so x1<Q+1<Q(Q+1). Let y=Q+1 and z=Q(Q+1). This attains x1 and excludes every smaller possible first denominator. Thus D(n)=x1. QED.

## Controls and limitations

For t=7, n=49, the construction gives (x,y,z)=(14,99,9702), while x0=13 is impossible. For t=49, n=2401, it gives (602,206487,42636674682). Seven declared integer cases and a negative control n=25, where x0=7 succeeds, are in raw_checks.json. These corroborate, rather than replace, the universal proof.

The two-term fraction criterion underlying the obstruction is KNOWN: Tieling Chen and Reginald Koo, *Two-term Egyptian fractions* (2013), Theorem 10(i). The proof above rederives the required exclusion directly. Priority of the precise extremal-family statement is unchecked. This result treats a composite family, not the hard prime cases, and establishes no global Erdős–Straus theorem or Lean verification.
