# Sharp two-addition/one-deletion local four-cycle count for odd projective orders

**E60-ODD-2SWITCH-2026-10-25:** ANALYTICAL_PROOF · PRIORITY_UNCHECKED · UNFORMALIZED.

**Theorem (precise, restricted scope).** For every odd prime power q≥3, there exists a simple graph G on n=q²+q+1 vertices, obtained by deleting one edge and adding two nonedges to the orthogonal polarity graph H of PG(2,q), such that

    e(G)=q(q+1)²/2+1,     #C4(G)=2q−4.

Here #C4 counts distinct noninduced, unoriented four-cycles. For odd prime powers q≥17, Füredi's theorem identifies e(H)=ex(n,C4), so G has exactly ex(n,C4)+1 edges. **This is not an unrestricted lower bound or a solution of Erdős #60.** In fact adding *one* edge to H already gives q−1 cycles, fewer than 2q−4 for q>3.

## Construction
Let F_q be a finite field of odd characteristic and define the nondegenerate symmetric bilinear form

    B(x,y)=x₀y₂+x₂y₀−2x₁y₁

on F_q³. Vertices of H are one-dimensional subspaces [x₀:x₁:x₂], with distinct vertices adjacent iff B(x,y)=0.

Write C_t=[1:t:t²] for t∈F_q and C_∞=[0:0:1]. These are precisely the q+1 absolute points. Distinct absolute points are pairwise nonadjacent because B(C_s,C_t)=(s−t)²≠0 and B(C_∞,C_t)=1.

Set a=[0:1:0], b=[1:0:−1]. Then ab is an H-edge, while the four absolute points C_∞, C₀, C₁, C_{−1} are distinct. Two simple H-paths are

    C_∞ — a — b — C₁
    C₀   — a — b — C_{−1}.

Define G=(H\{ab})∪{C_∞C₁,C₀C_{−1}}. These are exactly one deletion and two additions.

## Exact noninduced C4 count
Every projective polar line contains q+1 points. Every absolute point has degree q (excluding its loop), and every nonabsolute point has degree q+1. Hence H has q(q+1)²/2 edges. Any two distinct polar lines meet at precisely one projective point, so any pair of vertices has at most one common neighbor; H contains no C4.

For two distinct absolute vertices X,Y, there are exactly q−1 simple length-3 X-to-Y paths in H, so adding XY creates exactly q−1 four-cycles. For completeness: X has q possible first neighbors u; each polar intersection u^⊥∩Y^⊥ determines a unique next vertex v. Exactly one first neighbor, the common neighbor of X,Y, gives the invalid v=Y. None of the other choices has v=u, since the polar line of an absolute X is tangent to the absolute conic, so u cannot be absolute. Nor can v=X, since X,Y are nonadjacent.

These q−1 paths for fixed X,Y are pairwise edge-disjoint: sharing an end edge would produce two common neighbors of u,Y (or X,v); traversing a shared middle edge in opposite orientations would give two common neighbors of X,Y. Either contradicts C4-freeness. This is also the known absolute-endpoint case of He–Ma–Yang Lemma 2.8.

Both displayed paths use ab, so deleting ab destroys exactly one cycle from each new-edge family. No four-cycle can use both new edges because these added edges are disjoint and all four endpoints are absolute, with no H-edges joining them. H itself has none. Therefore

    #C4(G)=(q−1−1)+(q−1−1)=2q−4.

For odd q≥17, the extremal equality ex(q²+q+1,C4)=q(q+1)²/2 follows from Füredi, quoted explicitly as Theorem 1.1 in He–Ma–Yang.

## Literature and implication
He–Ma–Yang's published Lemma 2.8 and Section 8 **already prove** the local lower bound sq−s² in their large-even-q stability setting. We do not rebrand that known lower bound as a discovery. What this note establishes explicitly is attainment at s=2 in **every odd-prime-power order**, disproving a +1 strengthening of the local bound. It cannot settle the global odd-order supersaturation problem: it says nothing about graphs not close to H. Literature priority for this explicit construction remains unchecked.
