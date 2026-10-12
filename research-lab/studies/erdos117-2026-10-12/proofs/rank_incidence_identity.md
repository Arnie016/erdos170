# Rank-incidence constraint in binary rank-five, three-output maps
**Date:** 2026-10-12. **Validity:** ANALYTICAL_PROOF. **Literature:** PRIORITY_UNCHECKED. **Formalization:** UNFORMALIZED.

## Claim contract and scope
Let V=F₂⁵ and W=F₂³. Suppose B:V×V→W is alternating bilinear, its values span W, and its **global** radical {v:B(v,w)=0 for all w} is zero. For v≠0 define r(v)=rank(w↦B(v,w)) and nⱼ=#{v≠0:r(v)=j}. For nonzero λ∈W*, let B_λ=λ∘B. Let t be the number of the seven nonzero λ for which B_λ has rank two.

**Theorem:** 3n₁+n₂=7+6t. In particular n₂≡1 (mod 3) and n₂≥1. If t=0, (n₁,n₂) is one of (0,7), (1,4), (2,1).

**Falsifier:** one faithful spanning B violating the identity. **Competing explanation:** similar counts in samples might arise from orbit bias. **Acceptance:** a universal incidence proof, checked at boundary ranks; exact independent finite regression is supplementary. **Stop:** either complete proof or valid counterexample. This is a structural filter, not an assertion about a(G) versus ω(G).

## Proof
Because the values of B span W, each nonzero λ gives a nonzero alternating scalar form B_λ. The rank of an alternating form is even, including in characteristic two. Since dim(V)=5, every B_λ has rank two or four, with respective radical dimension three or one.

Double-count the set
I={(v,λ): v∈V\{0}, λ∈W*\{0}, λ(B(v,w))=0 for every w∈V}.

For fixed v≠0, the contraction T_v:w↦B(v,w) is nonzero because the **global** radical is zero, so 1≤r(v)≤3. Nonzero λ annihilating im(T_v) number 2^(3−r(v))−1. For contraction ranks 1,2,3 these counts are 3,1,0. Thus |I|=3n₁+n₂.

For fixed nonzero λ, the allowable v are precisely nonzero radical vectors of B_λ. A rank-two scalar form contributes 2³−1=7 vectors; a rank-four form contributes 2¹−1=1. There are t of the former and 7−t of the latter, so |I|=7t+(7−t)=7+6t. Equating counts proves the identity. Modulo 3, n₂≡1. At t=0, 3n₁+n₂=7 and n₂≥1 leaves exactly the stated three possibilities. QED.

## Applicability, invariance, limitations
For any group G with G/Z(G)≅F₂⁵ and G'≅F₂³, its commutator map B has these hypotheses: the quotient is central-by-abelian, squares are central, commutators have order at most two, images span the derived subgroup, and taking the **actual** central quotient kills the common radical. This includes infinite centers, but **not** general arbitrary groups. Basis changes in V and W do not affect the contraction ranks; a change of central coordinates permutes the seven nonzero λ and preserves t. No restriction-induced radical is used.

The result shows that at least one nonzero vector has contraction rank two. It does not imply χ(B)=ω(B), a chromatic gap, a new bound for h(n)=sup_{ω(G)≤n}a(G), a better uniform constant or a finite threshold for an eventual formula. A constructed abelian cover gives an upper bound for a(G); a noncommuting clique gives a lower bound. The previous complete rank-five **two-output** census cannot be extended to three outputs by this identity.

## Exact computation, limited purpose
A fixed 1,024-index deterministic SHA-256 sample yielded 1,007 faithful rank-three triples. An independent direct 5×5 matrix implementation checked the identity on every one, with a deliberately corrupted n₂ count rejected. Another sequential implementation built wedge-commutation graphs: 787 faithful triples had no isotropic 3-space; its exact-threshold search found an 11-clique for all 787. This only rejects the very narrow sufficient-gap candidate “no isotropic 3-space and ω≤10” **within that sample**. Neither sample is an exhaustive pencil classification, an independent human check, a Lean proof or a new global-bound certificate.

Worker 1 printed EARLY_STOP due to a cosmetic condition using full_rank==1024 rather than indices_processed==1024; its own log confirms all 1,024 indices processed, exit 0. Both workers ran with 35-second hard caps and exit code zero. The proof above does not depend on their output.
