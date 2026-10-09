# Central-basis shear and target saturation

Status: analytical counterexample to a coordinate-dependent local criterion. Literature priority unchecked. No general Erdős solution is claimed.

On V=X+Y+Z+<u,w> over F2, with dim X=dim Y=a and dim Z=b<=a, set

`beta0=sum x_i wedge y_i`, `beta1=u wedge w`, `beta2=sum_{i<=b} y_i wedge z_i`.

Replace beta0 by beta0'=beta0+beta2 while retaining beta1,beta2. This is the invertible central-coordinate map (c0,c1,c2)->(c0+c2,c1,c2), so B(v,w)=0 iff B'(v,w)=0. Applying the same map to the cocycle central coordinates yields an isomorphic presentation. Thus the commuting graph and abelian-cover problem are unchanged.

On the old slice T=Y_b+Z, beta0 originally vanishes, whereas beta0'|T=beta2|T has full rank 2b.

## An explicit surviving descent

The radical of beta0' is R0=<x_i+z_i:i<=b>+<u,w>. Pairing with X forces y=0; pairing with Y forces x_i=z_i for i<=b and x_i=0 for i>b. Hence dim R0=b+2 and rank beta0'=2a.

A0=R0+Y is isotropic, and A0/R0 has dimension a, so it is a Lagrangian child. On A0, beta1 has rank two. Its radical is R1=Y+<x_i+z_i:i<=b>. Choose the next isotropic child A1=R1+<u>.

Now beta2(y_i,x_j+z_j)=delta_ij, giving rank(beta2|A1)=2b. Also beta0'(x_j,y_i)=delta_ij, while beta0' vanishes on the radical graph directions. Therefore the source image of A1 has dimension a. The local pair (t,rho) remains (a,2b).

Thus target saturation in one named source coordinate is not an intrinsic obstruction. The old hard-family interpretation is conditional on its separate branch/scale hypotheses, but the basis change and the displayed ranks require no such asymptotic assumptions.

## Verification

`python3 src/check_coordinate_shear.py` checks all 36 pairs 1<=b<=a<=8. It verifies the source ranks, the proposed radical generators' rank, isotropy, the child dimensions, the surviving target rank and interaction rank. This is a finite matrix regression, not a Lean proof.
