# Coupled-padding rank budget

Historical analytical note, originally 2026-10-03. Literature priority unchecked. The accompanying program verifies the stated rank inequalities only in its declared finite domains. The asymptotic hard-family application retains its external project hypotheses; it is not a new global result for Erdős #117.

Let S,T be binary vector spaces of dimensions 2r,2b. Let sigma on S and tau on T be nondegenerate alternating forms. Consider source and target coordinates

`beta0((x,y),(x',y')) = sigma(x,x') + C(x,y') + C(x',y) + gamma(y,y')`,

`beta2((x,y),(x',y')) = tau(y,y')`,

where C:S->T* has rank q and gamma has rank 2u. Other coordinates may be present; the two displayed coordinates alone can certify noncommutation.

## Product-clique bound

An alternating form of rank R loses at most 2c rank when restricted to a codimension-c subspace: deleting c rows and c columns loses at most 2c matrix rank.

Choose a maximal gamma-isotropic L<=T. Its codimension is u, so tau|L has rank at least 2b-2u. It supplies a scalar clique of size 2b-2u+1 (use the zero vector for the singleton rank-zero case).

Let K={x in S:C(x,L)=0}. Its codimension is at most q, so sigma|K has rank at least 2r-2q. Choose a scalar clique of size max(1,2r-2q+1) in a nondegenerate summand of K. The two chosen summands intersect trivially as subspaces of S direct sum T.

For two distinct sums x+y and x'+y', if x differs from x', beta0 detects them because gamma vanishes on L and all cross terms vanish. If x=x', tau detects y!=y'. Consequently

`omega >= max(1,2r-2q+1) (2b-2u+1)`.

## Conditional asymptotic consequence

For the historical hard scaling t=a=mb, b=K0(m+1), K0=3*63^3, we have b=Theta(sqrt(t)). Suppose r/t->infinity while t>=c omega^(2/3) for a fixed c>0. Since q<=2b, q/r->0. If u<=(1-epsilon)b along an infinite subsequence, the bound gives omega=Omega(rb), hence t/omega^(2/3)->0, a contradiction. Therefore u/b->1.

This is a necessary condition in this fixed coordinate architecture, not a basis-invariant group obstruction. The following day's shear example shows that full source rank on a named target slice does not by itself destroy the local interaction. Couplings that also change beta2 or the target slice lie outside this statement.

## Verification

Run `python3 src/verify_coupling_rank_budget.py` from the study directory. Domain: every alternating gamma and every binary cross matrix for (r,b)=(2,1),(1,2), totaling 16,896 cases. The program verifies the two restriction-rank losses, not a general group or cover classification.
