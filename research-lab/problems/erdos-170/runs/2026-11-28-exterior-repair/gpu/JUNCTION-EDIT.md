# What happens when both junctions move to one mark?

Let $\delta(r)$ be the least number of positive distances missing from $[1,N]$ after deleting any two distinct original marks of the canonical $W(r,2r)$ and inserting a fresh integer in that interval. The [universal obstruction](../PROOF.md) shows that a perfect repair is impossible.

**Theorem.** For every $r\ge2$, deleting the two junctions and inserting $L+b$ misses exactly $r+5$ distances. In particular, $\delta(r)\le r+5$.

Write $a=2r+1$, $b=a+1$, $q=a+b$, $s=2r$, $T=sq$, $L=(r+1)a$, $R=L+T$, $F=(r+1)b$ and $H=R+F=N-r$. The original marks are the left cap $0,\ldots,r$, left progression $a,\ldots,ra$, middle progression $C_j=L+jq$ for $0\le j\le s$, right progression $R+b,\ldots,R+rb$ and right cap $H,\ldots,H+r$. The deleted junctions are $C_0=L$ and $C_s=R$. Since $0<b<q$, $z=L+b$ is fresh.

## Which distances disappear on deletion?

The following rows list all losses before adding $z$.

| Source of lost pairs | Missing positive distances |
| --- | --- |
| $L$ to the left and right caps | $L$, $T+F$ |
| $R$ to the left and right caps | $R-x,\ F+x$ for $0\le x\le r$ |
| $L$ to the right progression | $T+kb$ for $1\le k\le r$ |
| $R$ to the left progression | $T+ka$ for $1\le k\le r$ |
| Between middle marks | $T-q,\ T$ |

For completeness, the [collision table](../../2026-11-28-mixed-repair/PROOF.md#justification-of-the-collision-table) determines when an affected pair has another representation. Type $02$ links to exterior type $34$ on $[L-r,L-1]$. Type $24$ links to exterior type $03$ on $[H-L+1,H-L+r]$. These intervals preserve all distances from $L$ to the caps except $L$ and $H-L=T+F$ itself. The $R$-to-cap distances are outside those intervals because $T>r$. The $L$-to-left and $R$-to-right progression distances are short multiples of $a$ and $b$ with surviving exterior representations. The two opposite-progression rows are beyond those linked multiples. Each remaining affected mixed pair has a globally unique distance. Type $22$ is isolated from every other type. Its retained indices $1,\ldots,s-1$ have differences $1,\ldots,s-2$ in units of $q$. Consequently the rows are disjoint and contain $4r+6$ distances in total.

## Which ones does the new mark recover?

* The left cap gives $z-x=F+(r+1-x)$ for $0\le x\le r$. This restores $F+1,\ldots,F+r$. The extra distance $F+r+1$ lies below $T-q$ for $r\ge2$ and is outside the other rows.
* The left progression gives $b+ka$ for $1\le k\le r$. These distances are at most $L+1<F$ and are $1\bmod a$, so none equals $L$.
* A remaining middle mark $C_j$ gives $jq-b<T-q$. To hit $L$ one needs $q\mid L+b=ra+q$, impossible since $\gcd(a,q)=1$ and $0<r<q$. To hit a loss $F+y$ with $0\le y\le r$ one needs $q\mid(r+2)b+y$. Multiplying by two gives $q\mid r+2+2y$, since $2b=q+1$. This integer lies strictly between zero and $q$.
* The right progression gives $T+(k-1)b$ for $1\le k\le r$. These values restore $T,T+b,\ldots,T+(r-1)b$.
* The right cap gives $T+rb+y$ for $0\le y\le r$. Its first value restores $T+rb$. Because $rb=L-r-1$, the rest restore $R-r,\ldots,R-1$.

Thus $3r+1$ lost distances return. The exact complement after the edit is
$$
\{L,F,T-q,R,T+F\}\ \cup\ \{T+ka:1\le k\le r\}.
$$
It contains $r+5$ elements. At $r=1$ the left progression additionally supplies $T-q$ because $z-a=q$. The stated range matters.

The [direct checker](check_template.py) reconstructs both complements from the literal gaps and all surviving pairs through $r=100$. The [exhaustive search](README.md) checks the optimum over **all** legal edits only through $r=32$. The all-radius optimality of this construction remains conjectural.
