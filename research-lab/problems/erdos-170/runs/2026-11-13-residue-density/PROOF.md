# Sharp right-tail density for a fixed arithmetic-progression left tail

**Claim:** R170-AP-TAIL-DENSITY-2026-11-12. Derived 12 November; corrected second checker independently reproduced 13 November.

Let integers \(r\ge2\), \(a\ge1\), \(t\ge1\) satisfy \((r-1)a<t\). Let \(N\ge 2t-1\) and suppose a restricted complete ruler \(A\subseteq\{0,\ldots,N\}\) has precisely
\[
L=A\cap[0,t-1]=\{0,a,2a,\ldots,(r-1)a\}.
\]
Write the reflected right tail as \(R=\{N-b:b\in A\cap[N-t+1,N]\}\subseteq[0,t-1]\). Put
\[
m_s=1+\left\lfloor\frac{t-1-s}{a}\right\rfloor\quad(0\le s<a).
\]
Then
\[
\boxed{|R|\ge M(r,t,a):=\sum_{s=0}^{a-1}\left\lceil\frac{m_s}{r}\right\rceil.}
\]
This bound is attained **for coverage of the longest \(t\) distances alone**, but does not imply a complete ruler.

## Proof

A representation \(N-u=b-\ell\), \(0\le u<t\), requires \(\ell\le u<t\) and \(b\ge N-u\ge N-t+1\). Hence the longest-distance condition is exactly \([0,t-1]\subseteq L+R\).

For each residue \(s\pmod a\), the required offsets form the length-\(m_s\) chain \(s,s+a,\ldots,s+(m_s-1)a\). Each element of \(R\) in residue \(s\) covers at most \(r\) chain positions, because \(L\) contains only \(r\) multiples of \(a\). Other residues cannot cover that chain. So \(R\) needs at least \(\lceil m_s/r\rceil\) entries in each class, yielding the lower bound.

For equality, construct
\[
R_*=\bigcup_{s=0}^{a-1}\{s+jra:0\le j<\lceil m_s/r\rceil\}.
\]
Every selected element lies in \([0,t-1]\). For any target \(s+ka\) write \(k=jr+e\), \(0\le e<r\). Then \(s+ka=(s+jra)+ea\in R_*+L\), proving complete coverage of the declared offsets and attaining the bound.

Writing \(t=ad+b\) with \(0\le b<a\), equivalently
\[
M=b\lceil(d+1)/r\rceil+(a-b)\lceil d/r\rceil.
\]

## Concrete consequence and limits

For \(N=500,t=30,r=3\), with exactly three left-window marks \(\{0,a,2a\}\) and \(1\le a\le14\), the minimum possible number of right-window marks needed for these longest 30 distances is 12 when \(a=3\), 10 when \(a=5\), 14 when \(a=7\). The entire fourteen-value exact table is frozen in the acceptance test. A different left-window pattern invalidates this conditional bound. Nothing here proves that a 39-mark length-500 ruler exists or improves the asymptotic Erdős #170 constant.

This elementary additive-covering result is literature **PRIORITY_UNCHECKED**, with no new-global-theorem claim and no Lean formalization.
