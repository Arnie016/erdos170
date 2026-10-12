# Partial RREF chart check for three-coordinate pencils

Date: 2026-10-12. Validity: NUMERICAL_LEAD for the computation and ANALYTICAL_PROOF for the counting lemma. Literature: PRIORITY_UNCHECKED. No new all-group theorem.

## Exact scope and falsifier

Let V=F2^5. An alternating bilinear map B:V×V→F2^3 defines a graph on 31 nonzero vectors: distinct x,y are adjacent when B(x,y)≠0. Write omega(B) for the largest clique and chi(B) for the chromatic number. The active existence question is whether some **faithful** B has chi(B)>omega(B). A falsifier of equality is an explicit B with independently certified clique optimality and an exclusion of omega(B)-colourings. Two-coordinate dimension-five equality is an older checked repository result, not evidence for this stronger claim.

## Analytic chart lemma

Order the ten alternating-form coefficients as (01,02,03,04,12,13,14,23,24,34). Every 3-dimensional scalar-form subspace with RREF pivot columns (0,1,2) has a *unique* RREF basis with integer masks

    a = 1 + 8*(t&127)
    b = 2 + 8*((t>>7)&127)
    c = 4 + 8*((t>>14)&127),   0<=t<2^21.

For each t these rows are linearly independent, and each pivot row is zero in the other pivot columns. Their leading columns are distinct, hence they form an RREF basis. Uniqueness of RREF gives distinct row spaces for distinct t and conversely every row space with that pivot pattern occurs. Exactly 2^21=2,097,152 **scalar-form subspaces** occur in this chart. They are not claimed to be distinct group isomorphism types. Other RREF pivot patterns exist.

A central extension group on V×F2^3 can be built using a bilinear cocycle f with commutator B(v,w)=f(v,w)+f(w,v). Its product (v,z)(w,z')=(v+w,z+z'+f(v,w)) is associative. Its order is 256; its *actual* central quotient has dimension five only when the common radical of B is zero. Unfaithful maps cannot be advertised as groups having that quotient dimension. Pairwise commuting classes generate abelian subgroups, so for these graphs chi(B) is a cover upper bound and omega(B) a clique lower bound; the standard reduction is an ordinary argument, not a Lean certificate from this run.

## Bounded experiment

Two distinct C++17 implementations were compiled and run sequentially, with hard timeout of at most 35 seconds per worker, without GPUs or paid services.

Worker 1 scanned k=0,...,8191 with t=(73191+1048573*k) modulo 2^21. The odd multiplier makes the 8192 t values pairwise distinct. Graph edges used wedge-coordinate parity; maximum cliques used an exact colour-bounded branch-and-bound; omega-colourability used exhaustive DSATUR. Each returned colouring was edge-checked. Exit 0, wall 0.09s, 8192 completed with omega=chi. Clique histogram: 9:125, 11:356, 13:2805, 14:2089, 15:1779, 17:1038. Raw 8,192 JSONL records SHA256: 568fc9514d5104f468c09a22cd7146dd49398171ae845f5bd2f1fa9d1ca9d162. The records are preserved in the user's dated downloadable package, not included in this public repository commit.

Worker 2 reconstructed each alternating scalar form as a 5×5 binary matrix, used Bron-Kerbosch for maximum clique and an independently written DSATUR test. Its attempted domain was every t=0,...,2^21-1. The process did not produce a normal exit receipt because of a timeout. The last flushed checkpoint reads OK_PREFIX 491520 OVERLAP 4096: every t in [0,491520) was evaluated without a reported gap; the first program's results agree on 4096 overlapping values. Checkpoint log SHA256: 3bb73c4b4f9880862b254ab3129525e604c3c6b5bce627bba9c4f95509b8d651. No result beyond the last completed prefix is certified, and no full-chart histogram exists.

The 8,192-case sampling receipt did not preserve raw clique or colouring witness vectors. This is not a complete independently reconstructed finite certificate. The ordinary combinatorial equivalences and exact search algorithms are inspectable in the downloadable dated package. CI in this repository is **not** claimed to have checked this new data or these new programs.

## Decision

No gap was found within the reported finite domains, but they are not the entire chart and include nonfaithful cases. Neither the optimal uniform constant nor an eventual h(n) formula follows. A verified faithful chromatic gap would change the research direction; a further repeated partial scan without better certificates should instead prompt a GL(5,2)-invariant structural audit. The October 4 basis-shear counterexample separately invalidates a coordinate-dependent local rank obstruction and is not contradicted by these tests.
