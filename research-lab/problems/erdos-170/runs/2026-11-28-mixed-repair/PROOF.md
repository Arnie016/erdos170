# The mixed repair is impossible

For every integer $r\geq 1$, the edit in the [frozen statement](STATEMENT.md) leaves at least one distance in $[1,N]$ uncovered. The proof uses unique representations of lengths in the original ruler. The final section derives the collision table for this specialization.

## Counting the lost lengths

Use the frozen notation and write $H=N-r$, $R=L+sq$, $E=B_0\cup B_1\cup B_3\cup B_4$, and $J=B_2\setminus\{L,R\}$. The blocks are ordered and disjoint since $r<a$, $ra=L-a$, $\min B_3=R+b$, and $\max B_3=H-b$. Thus $|E|=4r+2$, $|J|=2r-1$, and $|W|=6r+3$. If the exterior deletion is $0$ or $N$, the distance $N$ is lost. Its only possible representing pair in the interval is $\{0,N\}$, and freshness forbids restoring a deleted endpoint.

Every pair with one mark in $J$ and the other in $E$ has a globally unique length, as checked below. Deleting $u\in J$ and $v\in E$ therefore removes the lengths of all pairs $\{u,e\}$ with $e\in E$ and $\{j,v\}$ with $j\in J$. The two sets of pairs intersect in exactly $\{u,v\}$. Their union consists of

$$
(4r+2)+(2r-1)-1=6r
$$

distinct lost lengths. After deletion there are $6r+1$ marks. Adding $z$ creates one pair with each surviving mark, so at most $6r+1$ of the missing lengths can be restored.

## Two additional losses

Write $ij$ for a pair between $B_i$ and $B_j$. In each row below, the two partners of $v$ give globally unique lengths. All partners lie outside $J$, so these edges were outside the first count.

| Exterior deletion $v$ | First partner and type | Second partner and type |
|---|---|---|
| Left cap $x\in[1,r]$ | $a$ ($01$) | $R$ ($02$) |
| Left progression $ia\in B_1$ | $H-b$ ($13$) | $R$ ($12$) |
| Right progression $H-ib\in B_3$ | $a$ ($13$) | $L$ ($23$) |
| Right cap $H+y$ with $0\leq y<r$ | $a$ ($14$) | $R$ ($24$) |

The following checks use the collision table below.

- For the left cap, $a-x\in[r+1,2r]$ is below the first positive $a$-multiple and every element of $T$. Also $R-x=L+sq-x>L-1$.
- For the left progression, $13$ has no link, and $R-ia\geq sq+a>ra$.
- For the right progression, $13$ has no link, and $H-ib-L\geq sq+b>rb$, using $H=R+(r+1)b$.
- For the right cap, $H+y-a=H-(a-y)$ with $0<a-y\leq a$ and every element of $T$ larger than $a$. Also $H+y-R=(r+1)b+y < sq+(r+1)b+1=H-L+1$.

These comparisons use $sq=2r(4r+3)$. The two partners in each row are distinct. Global uniqueness makes their lengths distinct from one another and from the first $6r$ losses. At least $6r+2$ lengths are lost, so one new mark cannot restore complete coverage. The argument treats each side of the ruler directly.

## Justification of the collision table

The consecutive gaps of $W$ are $r$ copies of $1$, then $r+1$, then $r$ copies of $a$, $s$ copies of $q$, $r+1$ copies of $b$, and $r$ copies of $1$. The block boundaries give the same gap sequence as the published Wichmann construction. Its completeness gives exactly $N$ positive lengths. [Saarela and Vanhatalo, Definition 16 and Theorem 18](https://www.combinatorics.org/ojs/index.php/eljc/article/download/v33i1p40/pdf/).

For each positive length $d$, consider the graph whose vertices are the pair types representing $d$. Use the links below whenever $d$ belongs to their length sets. Write $\mathcal A=\{a,2a,\ldots,ra\}$, $\mathcal B=\{b,2b,\ldots,rb\}$, $I=[L-r,L-1]=[rb+1,rb+r]$, and

$$
T=\{ib+y:1\leq i<r,\ r+1-i\leq y\leq r\}.
$$

For a set of lengths $S$, write $H-S=\{H-d:d\in S\}$. The final column gives a representation in each linked type. Two marks separated by a comma form one pair.

| Linked types | Lengths | Representing pairs in the two types |
|---|---|---|
| $00,44$ | $1\leq d\leq r$ | $(0,d)$ and $(H,H+d)$ |
| $01,34$ | $d=ib+y\in T$ | $(x,(i+1)a)$ and $(H-ib,H+y)$, where $x=a-i-y$ |
| $03,14$ | $d=H-(ib+y)\in H-T$ | $(y,H-ib)$ and $((i+1)a,H+x)$, with the same $x$ |
| $02,34$ | $d=rb+y\in I$, $1\leq y\leq r$ | $(r+1-y,L)$ and $(H-rb,H+y)$ |
| $03,24$ | $d=H-(rb+y)\in H-I$, $1\leq y\leq r$ | $(y,H-rb)$ and $(L,H+r+1-y)$ |
| $01,12$ | $d=ia\in\mathcal A$ | $(0,ia)$ and $((r+1-i)a,L)$ |
| $01,11$ | $d=ia$, $1\leq i<r$ | $(0,ia)$ and $(a,(i+1)a)$ |
| $23,34$ | $d=ib\in\mathcal B$ | $(R,H-(r+1-i)b)$ and $(H-ib,H)$ |
| $33,34$ | $d=ib$, $1\leq i<r$ | $(H-(i+1)b,H-b)$ and $(H-ib,H)$ |

Here $b=a+1$ and $L=rb+r+1$. For the two rows using $T$, $r+1-i\leq x\leq r$ and $(i+1)a-x=ib+y$. In particular each listed link joins actual representations.

The combined graph of the nine links is a forest. Its nontrivial components are the edge $00,44$, the star centered at $03$, and the tree with central edge $01,34$ and leaves $02,11,12,23,33$. For fixed $d$, if $m_d$ types are present, this forest contributes at most $m_d-1$ links. Its total number of links counted over lengths is $r^2+6r-2$, since $|T|=r(r-1)/2$ and the other seven rows contribute $7r-2$.

To see this bound is saturated, the total surplus of pairs over lengths is

$$
\binom{6r+3}{2}-(12r^2+14r+3)=6r^2+r .
$$

All types with distinct blocks except $04$ are injective as maps from pairs to lengths. A cap has diameter $r$, less than any of the progression steps. For two progressions the steps $a,b,q$ are pairwise coprime, and at least one of the index ranges has diameter $r-1$, smaller than the other step. Equality of lengths would force that index difference to be a multiple of the other step, hence zero. In a same-block type on $m$ equally spaced marks there are $m-1$ lengths, so its surplus is $\binom m2-(m-1)$. In type $04$ the lengths run through $H-r,\ldots,H+r$, giving surplus $(r+1)^2-(2r+1)=r^2$. Summing the within-type surplus gives

$$
r(r-1)+(r-1)(r-2)+s(s-1)/2+r^2=5r^2-5r+2 .
$$

Subtracting yields $\sum_d(m_d-1)=r^2+6r-2$, exactly the link count. Thus each length graph is connected. Any pair of an injective type with no incident link at its length is globally unique.

For the strict middle–exterior pairs, the only relevant links come from $02,12,23,24$. Write $u=L+jq$ with $1\leq j\leq s-1$. For $02$, the length is at least $L+q-r>L-1$, above $I$. For $24$, it is at most $N-L-q<H-L+1$, below $H-I$. For $12$, a linked length in $\mathcal A$ would imply $a\mid j$ since $\gcd(a,q)=1$, but then the length is at least $a+aq>ra$. For $23$, a linked length in $\mathcal B$ would imply $b\mid(s-j)$ since $\gcd(b,q)=1$, but then the length is at least $b+bq>rb$. This proves the uniqueness used in the first count.

The map $x\mapsto N-x$ preserves distances and transports the edit conditions to the reflected ruler. Its middle grid then begins at $N-R=L+a$, and the two exterior progressions swap sides. The proof treats the blocks at their original coordinates.
