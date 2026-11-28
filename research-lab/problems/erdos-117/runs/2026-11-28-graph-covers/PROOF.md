# Exact clique and cover numbers for graph alternating maps

For a finite simple graph $H$ with vertex set $X$ and a finite field $F$ of order $q$, put $V=F^X$. Choose an orientation of each edge and set $B_H(x,y)_{ij}=x_i y_j-x_j y_i$. Let $a(B_H)$ be the least number of totally $B_H$-isotropic linear subspaces covering $V$, and let $\omega(B_H)$ be the largest size of a set of pairwise nonorthogonal vectors. A support $S\subseteq X$ dominates $H$ when every vertex outside $S$ has a neighbor in $S$. Write $c(S)$ for the number of connected components of the induced graph $H[S]$, with $c(\varnothing)=0$.

**Theorem.** The two invariants are equal and satisfy

$$
  a(B_H)=\omega(B_H)=
  \sum_{\substack{S\subseteq X\\S\ \mathrm{dominates}\ H}}
          (q-1)^{|S|-c(S)}.
$$

In particular, for $q=2$ this is the exact number of dominating subsets, including in graphs with isolated vertices.

## Proof

Write $x^\perp=\{y:B_H(x,y)=0\}$ and $S=\operatorname{supp}(x)$. Let $N[S]$ be $S$ together with all its neighbors. For an edge from $i\in S$ to $j\notin S$, the condition of orthogonality forces $y_j=0$, because $x_i\ne0$ and $x_j=0$. For an edge internal to $S$, it forces $y_i/x_i=y_j/x_j$. These ratios are constant along each connected component of $H[S]$. Edges disjoint from $S$ impose no equations. Conversely, the described conditions make each edge determinant zero. Reversing the edge orientation negates a determinant and preserves its vanishing, in odd characteristic as well as in characteristic two.

Thus if $S$ dominates and its components are $C$, we have

$$
  x^\perp=\bigoplus_{C\text{ component of }H[S]}F\,x|_C,
$$

where each $x|_C$ is extended by zero off $C$. This is an isotropic linear space. Indeed, two members vanish off $S$, and their restrictions to the endpoints of any edge within $S$ are multiples of the same pair of coordinates of $x$. Different components have no edge between them. A component scalar can be zero, so the space also includes members with smaller supports.

Suppose $x,y$ both have dominating supports and $B_H(x,y)=0$. The formula for $x^\perp$ gives $\operatorname{supp}(y)\subseteq\operatorname{supp}(x)$. Since $B_H(y,x)=-B_H(x,y)=0$, the formula for $y^\perp$ gives the reverse containment. On their common support, the respective vectors differ by multiplication by a nonzero scalar on each component, and any such multiplication preserves orthogonality. Hence the orthogonality classes within the dominating-support vectors are exactly the choices of the support and one full-support line in $F^C$ for each component $C$. Fixing one coordinate to $1$ on every component counts $(q-1)^{|C|-1}$ choices there. Their product is $(q-1)^{|S|-c(S)}$.

Take one vector $x$ from each class and collect all the spaces $x^\perp$. The representatives form a clique. To see that the spaces cover, start with $y\in V$ of support $T$ and let $I$ be a maximal independent set of $H[X\setminus N[T]]$. Such a set exists by finite greedy extension, and every unselected vertex of that induced graph has a neighbor in $I$ by maximality. Therefore $T\cup I$ dominates $H$. The vector $u$ formed by retaining $y$ on $T$, placing $1$ on $I$, and placing zero elsewhere has dominating support. Each vertex of $I$ is a singleton component of $H[T\cup I]$, since it has no edge to $T$ or the rest of $I$. The vector $y$ is a componentwise multiple of $u$, with scalar one on components in $T$ and zero on $I$. Thus $y\in u^\perp$. The representative $x$ for $u$'s class satisfies $x^\perp=u^\perp$, so $y$ is covered.

If $T=\varnothing$, this construction chooses a maximal independent set in all of $H$ and covers the zero vector by setting every component scalar to zero. If $H$ has an isolated vertex, that vertex occurs in every dominating support and contributes a factor $1$ to the weight. The proof allows different scalars on different components, including isolated vertices. For an edgeless graph, $X$ is the only dominating support, every component is a singleton, and the formula gives $1$.

Each isotropic subspace contains at most one member of a pairwise nonorthogonal set. The clique and cover just constructed meet in size, so both are optimal. $\square$

For a complete graph on $n\ge 1$ vertices every nonempty support is connected. The count then reduces to $(q^n-1)/(q-1)$, the number of one-dimensional subspaces in $F^n$. For a disjoint union, a support is dominating precisely when its intersection with each factor is dominating. The exponent adds over the factors, so the count multiplies.

## Structure of all maximal isotropic subspaces

For an arbitrary isotropic subspace $U$, take $T$ to be the union of the supports of its members. On any edge within $T$, the two coordinate functionals on $U$ are nonzero and proportional. In fact, a member $v\in U$ with $v_i\ne0$ forces $w_j=(v_j/v_i)w_i$ for every $w\in U$, and $v_j\ne0$ follows because coordinate $j$ occurs somewhere in $U$. Propagating along each component gives a full-support line on that component. The space $U$ lies in the direct sum of these lines, each extended by zero. Even if no vector in $U$ has support $T$, that direct sum has a vector of support $T$.

Add singleton lines on a maximal independent set of the undominated vertices as above. The enlarged space is one of the constructed spaces $x^\perp$, so every maximal isotropic subspace must be one of those. Conversely, if an isotropic subspace contains $x^\perp$, it contains $x$ and every member is orthogonal to $x$. It is therefore equal to $x^\perp$. Distinct representatives yield distinct maximal spaces because equality of spaces would imply orthogonality of those representatives. Thus the weighted count also counts all maximal isotropic subspaces.

## Group realization and evidence scope

For each oriented edge $ij$, set $f(x,y)_{ij}=x_i y_j$. On $V\times F^{E(H)}$, the law

$$
  (x,z)(y,t)=(x+y,z+t+f(x,y))
$$

defines a finite group by bilinearity, with commutator $f(x,y)-f(y,x)=B_H(x,y)$ in the central second factor. The projection of an abelian subgroup to $V$ has an isotropic $F$-linear span by bilinearity, even when that projection is only additive. The preimage of an isotropic $F$-subspace is an abelian subgroup. Lifting vectors and projecting clique elements preserves noncommutation, so the group also has these exact clique and abelian-cover numbers. Its common radical is the span of the isolated vertex coordinates. Indeed, such coordinates appear in no edge, whereas a nonzero coefficient on an incident vertex is detected by pairing with the neighboring coordinate vector.

The local baseline handled the four three-edge graphs on five binary coordinates. The theorem extends that calculation to all finite simple graphs and fields. The unrestricted Erdős #117 profile concerns a supremum over all groups. Lecomte's preprint states a formula for that supremum and describes its extremizers. [Three-edge baseline](https://github.com/Arnie016/erdos170/blob/9e5616772dded2475ff7caf6ce9cbccc5ffd31cb/research-lab/studies/erdos117-2026-10-12/proofs/three_edge_coordinate_forms.md) · [Lecomte preprint](https://arxiv.org/html/2611.15068v1)

The rank-five three-output study at a separate pinned revision certifies 64 named scalar spans. The graph theorem requires a basis of edge forms in a common vertex coordinate system. General scalar spans require additional arguments. [Rank-five sample](https://github.com/Arnie016/erdos170/blob/5090494f778f3290881c2d3b63f67e2ba64279bf/research-lab/studies/erdos117-2026-10-14/README.md)

Tobias Rossmann's 2022 paper gives a direct precursor to the orthogonal-space calculation in Lemma 3.2. The dual of his specialized adjacency module is the space used above, and his proof splits the module using the support components and the coordinates outside the closed neighborhood. His Proposition 6.3 obtains the connected domination polynomial from a conjugacy-class counting coefficient. Proposition 7.5, attributed there to Matteo Cavaleri, uses dominating supports to compute the distinct extremal invariant that controls the degree of the class-counting polynomial. [Published paper](https://doi.org/10.1112/blms.12665) · [Rossmann, Lemma 3.2 and Propositions 6.3 and 7.5](https://arxiv.org/html/2107.05564v3)

Blatherwick gives the analogous centralizer description over integral exponents. Qiao counts all isotropic subspaces of the same graphical alternating matrix spaces by dimension, using proportional rows on induced-support components. The proof above derives the matching clique and cover formula from the component calculation. Literature priority for that formula remains unverified. [Blatherwick, §4](https://doi.org/10.1017/S0017089508004187) · [Qiao, §2](https://arxiv.org/html/2408.09963)

The binary theorem has a separate Lean 4.19.0 proof. The general finite-field argument, maximal-space classification and group realization above are ordinary proofs. See the [reproduction record](REPLAY.md).
