# Erdős #117: the first exact all-group threshold

**Validity:** ANALYTICAL_PROOF. **Literature:** PRIORITY_UNCHECKED (related three-subgroup structure is classical). **Formalization:** UNFORMALIZED. No new global asymptotic bound.

## Statement, quantifiers and falsifier

Let a(G) be the least cardinality of an abelian-subgroup cover, omega(G) the largest size of a pairwise noncommuting set, and h(n)=sup{a(G):omega(G)<=n} over **all groups**, including groups with infinite center.

**Theorem.** For any nonabelian group G, omega(G)=3 if and only if G/Z(G) is Klein four. In either case a(G)=3. Consequently h(1)=h(2)=1 and h(3)=3.

A group with omega=3 and non-Klein central quotient, or an abelian cover needing more than three members, would falsify the claim. The plausible competing explanation is that the common intersection of three centralizers could be nonabelian; the proof explicitly rules that out.

## Elementary proof, including infinite groups

Choose noncommuting x,y in G. Then x,y,xy are pairwise noncommuting, so omega(G)>=3. Assume omega(G)=3 and define A=C_G(x), B=C_G(y), C=C_G(xy). Any g outside A union B union C would make x,y,xy,g a noncommuting 4-set. Thus G=A union B union C. Each subgroup is proper, and the union is irredundant: x lies only in A, y only in B, and xy only in C.

Set N=A intersect B intersect C. We claim A intersect B=A intersect C=B intersect C=N. If t lies in A intersect B but not C, then t(xy) lies outside A union B (because xy does), hence lies in C, forcing t into C. Contradiction. The other cases are symmetric.

Each of A,B,C has exactly two N-cosets. For a,a' in A outside N and b=y in B outside N, the products ab and a'b lie in C, because neither can belong to A or B. Hence aa'^{-1}=(ab)(a'b)^{-1} belongs to A intersect C=N. Thus A=N union xN; similarly B=N union yN and C=N union xyN. The four cosets N,xN,yN,xyN partition G. N has index two in each of A,B,C, so N is normal in every one of them and therefore in G.

**Key hinge: N is abelian.** If u,v in N did not commute, then u,v commute with x,y and xy, and the five elements
x, y, (xy)u, (xy)v, (xy)uv
would be pairwise noncommuting. The first two are noncommuting with each other and with each of the last three; among the last three, multiplying by the common element xy does not alter the noncommuting relations of u,v,uv. Contradiction to omega(G)=3. So N is abelian.

Now N commutes with x,y by its definition and with itself by abelianness; every element of G is in N,xN,yN or xyN. Therefore N is central. Since Z(G) lies in all three centralizers, N=Z(G). The quotient G/Z(G) has four elements represented by 1,x,y,xy, each nonidentity coset of order two, so it is C2 x C2.

The groups A=N union xN, B=N union yN and C=N union xyN are abelian and cover G, giving a(G)<=3. The three noncommuting elements x,y,xy yield a(G)>=3. Thus a(G)=3.

Conversely, if G/Z(G)=C2 x C2, G is nonabelian (a group generated modulo its center by commuting lifts would be abelian). No noncommuting set can contain two members of one central coset, so omega(G)<=3. Nonabelianity yields omega(G)>=3, and the previous argument gives a(G)=3. Abelian groups have omega=a=1, and D8 exhibits omega=a=3, proving h(1)=h(2)=1 and h(3)=3. QED.

## Checks, provenance and limitation

Independent exact Cayley-table/subgroup-cover controls gave (omega,a,|G/Z|): D8=(3,3,4), Q8=(3,3,4), S3=(4,4,6), C4=(1,1,1). These are finite controls, **not** the proof for arbitrary groups. The proof itself uses no external theorem. See check_small_groups.py and RUN.md for execution receipt.

The three-subgroup quotient phenomenon is classical (Scorza); this is a readable all-group baseline, not a novelty claim. It changes no bound for unrestricted n. The next discriminating problem is whether h(4)=4. A group with omega<=4 and a>=5 would falsify that equality, but its cover lower bound and clique upper bound must both be certified.