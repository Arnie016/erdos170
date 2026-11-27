# Arithmetic-progressions at tight sparse-ruler endpoints

**Claim:** R170-TAIL-AP-2026-11-11  
**Validity:** ANALYTICAL_PROOF, with a separate EXACT_FINITE_CERTIFICATE  
**Literature:** established polynomial-tiling method, specific formulation PRIORITY_UNCHECKED  
**Formalization:** UNFORMALIZED

## Precise statement

Let integers \(r\ge2,\ q\ge1,\ a\ge1\), put \(t=rq\), and assume \(N\ge2t-1\). Let \(A\subseteq\{0,\ldots,N\}\) measure **every** distance \(1,\ldots,N\). Suppose its leftmost \(t\)-window is exactly

\[
L=A\cap\{0,\ldots,t-1\}=\{0,a,2a,\ldots,(r-1)a\},
\]

while its reflected rightmost \(t\)-window

\[
R=\{N-b:b\in A,\ N-t+1\le b\le N\}
\]

has cardinality \(q\).

**Theorem.** Necessarily \(a\mid q\), and \(R\) is uniquely forced:

\[
R=\bigcup_{j=0}^{q/a-1}\{raj,raj+1,\ldots,raj+a-1\}.
\]

Conversely, these prescribed tails realize each of the *longest* \(t\) distances exactly once when \(a\mid q\). This does **not** guarantee that the full ruler covers smaller distances.

## Ordinary proof

If \(N-s=b-u\) for \(0\le s<t\), with both \(b,u\in A\), then \(u\le t-1\) and \(b\ge N-t+1\). Thus \(s=u+(N-b)\) lies in \(L+R\). The windows are disjoint because \(N\ge2t-1\).

There are exactly \(|L||R|=rq=t\) ordered sums and \(t\) required values \(0,1,\ldots,t-1\). Therefore every one occurs exactly once, with none outside that range:

\[
(1+X^a+\cdots+X^{(r-1)a})R(X)
=1+X+\cdots+X^{rq-1}.
\]

Choose \(\zeta\), a primitive \(ra\)-th root of unity. Since \(\zeta^a\) is a primitive \(r\)-th root, the left factor vanishes. Since \(\zeta\ne1\), the geometric sum on the right vanishes only if \(\zeta^{rq}=1\). Its order is \(ra\), hence \(ra\mid rq\), so \(a\mid q\).

Write \(q=ah\). Polynomial division uniquely gives

\[
R(X)=\frac{1-X^{rq}}{1-X}
\cdot\frac{1-X^a}{1-X^{ra}}
=(1+X+\cdots+X^{a-1})
(1+X^{ra}+\cdots+X^{ra(h-1)}).
\]

The nonzero coefficients yield exactly the blocks claimed. Conversely their pairwise sums with \(L\) cover \(0,\ldots,t-1\) once each. QED.

## A concrete endpoint-density exclusion at length 500

Counting the longest \(t=rq\) distances only forces \(q\) reflected right-tail marks from an \(r\)-mark left tail. The theorem shows that **if \(a\nmid q\), at least \(q+1\) right-tail marks are necessary**, conditional on the exact left-tail shape.

For \(N=500,\ r=3,\ q=10,\ t=30\): if positions \(0,\ldots,29\) contain exactly \(\{0,a,2a\}\) and positions \(471,\ldots,500\) contain exactly ten marks, then \(a\in\{1,2,5,10\}\). Other allowed offsets \(1\le a\le14\) require at least 11 right marks. No specific length-500 construction was checked here.

## Finite challenge

Freeze \(r=3,q=6,t=18,N=35\). For each \(a=1,\ldots,8\), inspect every six-subset of reflected positions \(0,\ldots,17\) containing zero: \(8\binom{17}{5}=49,504\) profiles. Two different exact algorithms accepted only four, with \(a=1,2,3,6\) and the forced tails. Both rejected a corrupted witness. The 49,504-profile result is an **exact finite certificate**, not the proof of the all-parameter theorem.

The original 16 October 39-mark length-500 near-ruler reportedly misses only distance 493 and resists every one-mark swap. Its actual mark list, in a previously named candidate.json, could not be recovered in this run. Therefore no two-swap classification or improvement in \(F(500)\) is claimed.

![An example of exact tiling and an illustrative failed pattern](multimark-tail.svg)
