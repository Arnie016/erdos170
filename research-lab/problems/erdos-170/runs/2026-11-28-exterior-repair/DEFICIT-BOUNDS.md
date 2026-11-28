# A linear deficit for every two-delete, one-insert edit

For the canonical $W(r,2r)$, let $\delta(r)$ be the minimum number of positive distances in $[1,N]$ left uncovered after any two distinct original marks are deleted and one fresh integer is inserted within the original interval. Endpoints and both junctions are eligible for deletion.

**Theorem.** For all $r\ge2$,
$$
\max(1,r-8)\ \le\ \delta(r)\ \le\ r+5.
$$
In particular, $\lim_{r\to\infty}\delta(r)/r=1$. Thus even the best repair has a deficit that grows linearly. The additive constant in the optimum remains undetermined.

Write $s=2r$, $a=2r+1$, $b=a+1$, $q=a+b=4r+3$, $L=(r+1)a$ and $R=L+sq$. The five blocks are the two caps, the two outer progressions of length $r$ and steps $a,b$, and the middle progression $C_j=L+jq$ ($0\le j\le s$). An exterior mark is in one of the four blocks outside the middle progression. The strict middle excludes $L,R$. The [block definitions](../2026-11-28-mixed-repair/STATEMENT.md) and [collision proof](../2026-11-28-mixed-repair/PROOF.md#justification-of-the-collision-table) give the uniqueness facts used below.

## Counting a column of lost distances

Fix a deletion and the new position $z$. Each globally unique pair distance incident to a deleted mark disappears before insertion. It returns only if a surviving mark is that distance from $z$. A *column* below consists of distances to the marks of one fixed progression. All distances in the column have one residue modulo its step when the original mark was outside the progression's interval.

For a fixed residue $d\bmod h$, a partner of $z$ must lie in one of $z\pm d\bmod h$. In each other block the residues are distinct, so at most two partners per block, or **eight outside the progression**, can restore that column. This holds for any of the three steps. For $h=q$ the exterior steps are coprime to $q$. For $h=a,b$, the remaining progression steps are $\pm1\bmod h$ and their block lengths are at most $h$. Each cap has diameter $r<h$. Each partner creates at most one distance. [Residue lemmas](../../../../lean/OpenMath/ExteriorRepair.lean)

Here is the constraint imposed by a *single* restoration through the progression itself. Let $P$ be the right progression of step $b$, and let the deleted mark $t\le L$ have unique distances to all of $P$. If any such distance returns using a partner in $P$, then $z<\min P$ and $z=t+kb$ with $|k|\le r-1$. Indeed for $z\ge\min P$ every possible distance to a mark of $P$ is at most $rb+r$, below the smallest required distance $sq+b$. Once $z,t$ are on the same side, equality of distances makes $z-t$ a difference of two marks of $P$. For the left progression of step $a$ and $t\ge R$, the corresponding bounds are $ra$ and $sq+a$, and $z=t+ka$.

The same argument works for a strict middle mark $t=C_j$ if we select the farther outer progression. Choose the right one when $j\le r$ and the left one otherwise. The minimum required distance is at least $rq+b$ on the right or $rq+a$ on the left. These exceed the same wrong-side maxima. Both junctions also have their respective far progression. The [collision proof](PROOF.md) establishes that each of these far columns is globally unique.

## Lower bound for every deletion pair

Let $m$ be the total number missing after an arbitrary edit.

1. **Two exterior marks.** Each loses a distinct column of $2r-1$ unique strict-middle distances, with one residue modulo $q$. If one column receives no middle partner, $m\ge2r-9$. Otherwise both residues equal $\pm(z-L)$ modulo $q$ because every distance from $z$ to a middle mark has one of those residues. Both columns then share the same at-most-eight exterior partners. At most $2r+1$ middle partners can be used in total, hence $m\ge4r-2-(2r+1)-8=2r-11$. For $r\ge3$ both estimates imply $m\ge r-8$.
2. **Two strict middle marks.** Each loses all $4r+2$ unique distances to exterior marks, and the two sets of losses are disjoint. There are only $6r+1$ surviving partners of $z$, so $m\ge2r+3$.
3. **A strict middle mark and an outer progression mark $v$.** The unique edges from the deleted middle mark to all exterior marks and from $v$ to all strict middle marks give $6r$ different losses, with their common edge counted once. In addition $v$ loses $r$ unique distances to the opposite outer progression and one to the opposite junction. All those pairs are distinct from the first $6r$. Thus $m\ge (7r+1)-(6r+1)=r$. Uniqueness of the extra distances follows from type $13$ and the unlinked long distances of types $12$ and $23$. The [formal modular-separation lemmas](../../../../lean/OpenMath/DeficitBounds.lean) give a second way to exclude simultaneous restorations through the common opposite progression in this case.
4. **A junction with a strict middle mark or a progression mark on the junction's side.** Both deletions have unique columns into the same opposite progression $P$ of step $h$. If the junction column has no $P$ restoration, $m\ge r-8$. Otherwise $z$ and both deleted marks lie on the same side of $P$, and $z$ has the junction's residue modulo $h$. A $P$ restoration for the other column would also give the companion's residue. These residues differ. The displacement to a strict middle mark is $kq$ with $1\le k\le 2r-1<h$. The displacement to a same-side progression mark is $k$ times the other outer step with $1\le k\le r<h$. Both steps are coprime to $h$. Hence one column has at least $r-8$ losses.
5. **A junction with the other junction or with an opposite-side progression mark.** The left deleted mark has a unique column into the right progression, and the right deleted mark has one into the left progression. The two deleted marks are at least $sq$ apart. If both columns receive restorations through their respective progression, $z$ must be within $(r-1)b$ of the left mark and $(r-1)a$ of the right. Those radii sum to $(r-1)q<sq$. One column therefore misses at least $r-8$ lengths.
6. **A cap and any middle mark, strict or junction.** Use the far progression for the middle mark. If it supplies no restoration, $m\ge r-8$. Otherwise $z\equiv L\pm ka\bmod q$ with $|k|\le r-1$, since $b\equiv-a\bmod q$. The cap mark has residue $x$ with $0\le x\le r$, because the right cap starts at $N-r=(3r+1)q$. A middle partner restoring even one of the cap's $2r-1$ unique strict-middle losses would force $z-L\equiv\pm(x-L)\bmod q$. Substituting, multiplying by two and using $2a=q-1$ would make $2x+r+1\pm k$ a multiple of $q$. It lies in $[2,4r]$, strictly between zero and $q$. Thus at most eight exterior partners restore this column, and $m\ge2r-9\ge r-8$.

The cases exhaust all unordered pairs. At $r=2$ the numerical inequality $m\ge r-8$ is automatic. The separate [no-perfect-repair theorem](PROOF.md) supplies $m\ge1$ throughout. The [two-junction construction](gpu/JUNCTION-EDIT.md) supplies the upper bound for every $r\ge2$. Dividing by $r$ and taking the limit proves the formula.

## What remains open

The [exhaustive finite minima](gpu/README.md) equal $r+5$ for $2\le r\le32$, including eight radii predicted before testing. A single legal edit missing at most $r+4$ distances at any larger radius would refute the sharp conjecture. One decisive proof target is to show that every competing edit misses at least $r+5$ distances. The theorem here quantifies the stability of this particular ruler under this particular edit at fixed length. It leaves the asymptotic optimum over all sparse rulers undetermined.

The proof of the quantitative bound is ordinary mathematics. Lean checks the residue lemmas, the modular-separation lemmas, and the rational epsilon implication from the geometric lower and upper bounds to the limit. [Lean declarations](../../../../lean/OpenMath/DeficitBounds.lean) The experiments and agent review are reproducible project evidence. Literature priority and independent human review remain open.
