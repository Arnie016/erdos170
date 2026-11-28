# Two deletions and one insertion in a Wichmann ruler

**Theorem.** For every integer $r\ge1$, remove any two distinct marks from the canonical ruler $W(r,2r)$ and add one integer from $[0,N]\setminus W$. The resulting set misses a distance in $[1,N]$. In particular, the two original marks can both be exterior.

Use the definitions in the [frozen statement](../../../../benchmarks/erdos-duel/PROJECT_BRIEF.md). Write $C_j=L+jq$, $J_-=C_0$, $J_+=C_{2r}$, and $H=N-r$. The five ordered blocks have indices $0$ through $4$, so the left and right outer progressions are $B_1$ of step $a$ and $B_3$ of step $b$. A distance is *unique* when exactly one original unordered pair realizes it. The following properties of the original ruler are proved in the [collision accounting](../2026-11-28-mixed-repair/PROOF.md#justification-of-the-collision-table).

1. All distances from strict middle marks $C_1,\ldots,C_{2r-1}$ to exterior marks are unique.
2. Each pair in $B_1\times B_3$, $\{J_-\}\times B_3$, or $B_1\times\{J_+\}$ has unique distance.

For property 2, type 13 is isolated. Type 23 has possible links only at positive multiples of $b$ through $rb$, whereas $p-J_-\ge2rq+b$ for $p\in B_3$. Type 12 has possible links only at positive multiples of $a$ through $ra$, whereas $J_+-p\ge2rq+a$ for $p\in B_1$. All three mixed types are injective. Consequently, deleting either member of any such pair loses its distance. Different such pairs lose different distances.

## A residue bound for an inserted mark

Fix an insertion $z$. If $|p-z|\equiv d\pmod h$, then $p\equiv z+d$ or $p\equiv z-d\pmod h$. A block whose members have distinct residues modulo $h$ can therefore supply at most two partners $p$ for a prescribed distance residue, including when those two residues coincide.

For $h=q$, each of the four exterior blocks has distinct residues. The caps have diameter $r<q$, and $a$ and $b$ are both coprime to $q$ since $b=a+1$ and $q=a+b$. For $h=a$ or $b$, let $P$ be the outer progression of that step. Each of the other four blocks has distinct residues modulo $h$. The caps have diameter $r<h$. The other outer step and $q$ are each $1$ or $-1$ modulo $h$, and the longest progression in question has $2r+1\le h$ terms. Thus for a distance column in one residue modulo the step of $P$, at most **eight** partners outside $P$ can contribute. The same bound holds outside $B_2$ for a column modulo $q$.

## Two exterior deletions

Each deleted exterior mark $x$ loses $2r-1$ distances to the strict middle marks. Their common residue modulo $q$ is $L-x$ on the left and $x-L$ on the right. Two distinct deleted marks lose $4r-2$ globally distinct distances. At most $16$ can be supplied by exterior partners of $z$, even if the two column residues coincide. All middle partners together supply at most $2r+1$. Hence a repair would require

$$4r-2\le16+(2r+1),$$

which is impossible for $r\ge10$. A deleted endpoint immediately loses $N$, since both $0$ and $N$ are required to realize it.

## When a junction is deleted

Take $r\ge17$. For the left junction, put $P=B_3$ and $h=b$. For the right junction, put $P=B_1$ and $h=a$.

**Localization.** If a deleted mark $t\le J_-$ has unique distances to all of $P=B_3$, then a repair requires $z=t+kh$ for an integer $|k|\le8$. The same holds for $t\ge J_+$ with $P=B_1$.

Indeed, the $r$ required distances form an arithmetic progression of step $h$. At least $r-8>0$ must be supplied by partners in $P$. For the right progression, if $z\ge\min P$, then every $|z-p|$ is at most $N-\min P=rb+r<2rq+b\le\min_{p\in P}|t-p|$. For the left progression, if $z\le\max P$, every $|z-p|$ is at most $ra<2rq+a\le\min_{p\in P}|t-p|$. Thus $t$ and $z$ lie on the same side of $P$ in both cases. Their two distance progressions are translates of each other. An intersection forces $z-t=kh$, and its size is $\max(0,r-|k|)$. Using all original partners of $P$ only enlarges the possible intersection. This proves the claim.

Write $J$ for a deleted junction. The following cases exhaust its possible companions.

* **A strict middle mark, or an outer progression mark on the same side as $J$.** Both deletions lose all $r$ distances to the *same* opposite progression $P$. The $2r$ distances are distinct. Each of the two columns can receive at most eight partners outside $P$, and $P$ has at most $r$ partners altogether. This would give $2r\le r+16$.
* **The other junction, or an outer progression mark on the opposite side.** The left deleted mark lies at or left of $J_-$, has unique distances to all of $B_3$, and localizes $z$ within $8b$. The right deleted mark lies at or right of $J_+$, has unique distances to all of $B_1$, and localizes $z$ within $8a$. The marks are separated by at least $2rq>8(a+b)$, so these intervals are disjoint.
* **A cap mark $v$.** Write $v\equiv x\pmod q$ with $0\le x\le r$, using $H=(3r+1)q$ for the right cap. Junction localization gives $z\equiv L+ka\pmod q$ with $|k|\le8$, since $b\equiv-a\pmod q$. If an edge from $z$ to a middle partner supplies even one of the cap mark's strict-middle losses, then $z-L\equiv\pm(x-L)\pmod q$. Thus $x\equiv L\pm ka\pmod q$. Multiplying by two and using $2a=q-1$ shows that $2x+r+1\pm k$ is a multiple of $q$. Both choices lie strictly between zero and $q$ because $r\ge17$ and $0\le x\le r$. Hence all $2r-1$ losses require partners outside the middle block, which can supply at most eight for this residue. The cap congruence argument was contributed by the independent referee in response to the desk's localization proof.

Two strict middle marks lose $2(4r+2)>6r+1$ unique distances to exterior marks, more than the insertion can create. One strict middle mark and one exterior mark are covered by the [mixed deletion proof](../2026-11-28-mixed-repair/PROOF.md).

## Finite boundary and scope

The remaining instances are $1\le r\le9$ for exterior pairs and $1\le r\le16$ for arbitrary pairs. For each deleted pair, form the bit mask $M$ of surviving marks. The mask of their nonnegative differences is the union of $M\mathbin{\gg}p$ over surviving $p$. Initialize candidate bits to the original complement inside $[0,N]$. For each missing $d$, intersect those candidates with $(M\mathbin{\ll}d)\mathbin{\vert}(M\mathbin{\gg}d)$. A candidate survives exactly when every missing difference has a surviving partner at distance $d$ from it.

The [integer-set verifier](check_repair.py) finds zero repairs for every pair through $r=25$, including all 29,016 pairs through $r=16$. Its masks represent all legal fresh positions simultaneously, and it short-circuits as soon as the candidate mask becomes empty. Its positive control deletes $1,2$ from $\{0,1,2,3,5\}$ and recovers exactly the fresh point $4$. A separate literal-pair implementation checks all endpoint-containing subsets of $[0,n]$ with more than two marks for $3\le n\le7$. Replay with `python3 check_repair.py` and compare the [per-radius counts](finite-check.txt). The referee independently implemented exhaustive boundary checks.

The [Lean 4.19 bridge](../../../../lean/OpenMath/ExteriorRepair.lean) proves the signed-partner residue restriction, the two-partner bound for a residue-injective block, and the final forbidden modular interval. The universal proof also uses the ordinary collision argument and the finite verification above. The theorem concerns this one-deletion-net local edit in the stated canonical family. The asymptotic optimization problem remains open.
