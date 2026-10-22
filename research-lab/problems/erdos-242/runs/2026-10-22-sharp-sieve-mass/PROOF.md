# Exact leading constant for Dahan's coprime sieve mass

**Claim E242-W-LEADING-20261022.** For positive integer J tending to infinity,
\[
W(J):=\sum_{\substack{1\le u,a\le J\\\gcd(u,a)=1}}\frac1{\varphi(4ua)}
 =\frac9{2\pi^2}(\log J)^2+O(\log J).
\]
This is a scoped analytical result about a sieve input, NOT a proof of the Erdős–Straus conjecture. Literature priority is unchecked.

## Ordinary proof

Set f(u,a)=1_{(u,a)=1}/φ(4ua). In the two-variable Dirichlet series
\[
F(s,t)=\sum_{u,a\ge1}f(u,a)u^{-s}a^{-t},\quad \Re s,\Re t>0,
\]
write α_p=p^{-1-s} and β_p=p^{-1-t}. At odd primes,
\[
F_p=1+\frac p{p-1}\left(\frac{\alpha_p}{1-\alpha_p}
                    +\frac{\beta_p}{1-\beta_p}\right).
\]
At p=2, accounting exactly for the fixed factor 4 in the totient,
\[
F_2=\frac12\left(1+\frac{\alpha_2}{1-\alpha_2}
                    +\frac{\beta_2}{1-\beta_2}\right).
\]
In each factor the coprimality condition forbids positive exponents in both u and a.

Factor F(s,t)=ζ(1+s)ζ(1+t)H(s,t). Multiplying the local series by
(1−α_p)(1−β_p) yields
\[
H_2=\frac12(1-\alpha_2\beta_2),\qquad
H_p=1+\frac{\alpha_p+\beta_p-(p+1)\alpha_p\beta_p}{p-1}\quad(p>2).
\]
The coefficients of H_p for p>2 at the exponent pairs (0,0),(1,0),(0,1),(1,1) are respectively 1, 1/[p(p−1)], 1/[p(p−1)], −(p+1)/[p²(p−1)]. The coefficients of H_2 are 1/2 at (0,0) and −1/8 at (1,1). For any 0<δ<1/2 the weighted total variation of the local coefficients is 1+O(p^{−2+2δ}). The Euler product therefore defines a two-variable Dirichlet series
\(H=\sum_{d,e\ge1}h(d,e)d^{-s}e^{-t}\)
with
\[
\sum_{d,e}|h(d,e)|(de)^\delta<\infty.
\]
In particular, its logarithmic moments are absolutely finite.

At s=t=0 we obtain
\[
H_2(0,0)=3/8,\qquad H_p(0,0)=1+p^{-2}\quad(p>2).
\]
Using the absolutely convergent Euler product,
\[
H(0,0)=\frac38\prod_{p>2}(1+p^{-2})
       =\frac38\frac{\zeta(2)/\zeta(4)}{1+1/4}
       =\boxed{\frac9{2\pi^2}}.
\]

The coefficients of ζ(1+s)ζ(1+t) are 1/(ua), hence the exact
two-dimensional Dirichlet convolution gives
\[
W(J)=\sum_{d,e\le J}h(d,e)
\,\mathcal H_{\lfloor J/d\rfloor}
\,\mathcal H_{\lfloor J/e\rfloor},
\quad\mathcal H_m=\sum_{r=1}^m1/r.
\]
Uniformly over d≤J, \(\mathcal H_{\lfloor J/d\rfloor}=\log J-\log d+O(1)\).
The finite logarithmic moments of h show that replacing the product
of harmonic numbers by (log J)^2 costs O(log J) after summation. The
weighted tail bound \(\sum_{\max(d,e)>J}|h(d,e)|=O(J^{-\delta})\)
permits replacing the truncated sum of h by H(0,0), with error
O(J^{-\delta} log² J). This proves the claimed asymptotic. QED.

## Adversarial checks and limitations

The even-prime factor 3/8 is essential. Dropping gcd(u,a)=1 changes
the product. The accompanying Python checker compares exact rational
totient values and reconstructed convolution values on explicitly
declared finite domains, then reproduces the numerical values printed
by Dahan. These checks are independent regression evidence, not the
universal proof.

Dahan (2026), Lemma 4.22 used the lower coefficient 3/(2π²), and
Section 7 explicitly asked for a lower coefficient at least 0.45.
The constant proved here equals 0.45594532639052. Substitution into
the subsequent prime-sieve estimates has NOT been audited and remains
conditional. No claim of a new all-n Egyptian fraction representation,
formal Lean proof or established literature priority is made.
