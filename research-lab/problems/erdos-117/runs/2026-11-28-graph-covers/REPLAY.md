# Reproduction record

The group desk starts from the match brief at `81ef55af285a4ea7db72a11e2a0b2a128be394fb`. Its ordinary proof establishes the binary target and the weighted formula for every finite field.

## Pinned binary formalization

The declaration `OpenMath.GraphCovers.binary_clique_cover_domination` ranges over an arbitrary vertex type with an exhaustive finite vertex list and a simple graph with Boolean adjacency, symmetry and no loops. A Boolean vector is a support's indicator function, addition is pointwise XOR and scalar multiplication is pointwise AND. Edge determinants vanish exactly when the two Boolean products in `orthogonal` agree. The result gives a duplicate-free list of exactly all dominating supports, a proof that it is a maximum clique, and an equal-length cover of the entire vector space by isotropic linear subspaces. Every competing cover has at least that length.

A fresh run staged the pinned Lake project from the frozen commit, copied in `OpenMath/GraphCovers.lean` and imported it from `OpenMath.lean`. In the final project with that import, run from the repository root:

```sh
cd research-lab/lean
lake env lean --version
lake build
lake env lean OpenMath/GraphCovers.lean
```

Observed exit status was zero for both build and audit. The Lake project uses Lean 4.19.0, compiler `6caaee842e94`, and Std.

```text
Built OpenMath.GraphCovers
Build completed successfully.
'OpenMath.GraphCovers.dominating_orthogonal_eq' depends on axioms: [propext, Quot.sound]
'OpenMath.GraphCovers.dominating_orthogonal_isotropic' depends on axioms: [propext]
'OpenMath.GraphCovers.exists_dominating_orthogonal' depends on axioms: [propext, Classical.choice, Quot.sound]
'OpenMath.GraphCovers.binary_clique_cover_domination' depends on axioms: [propext, Classical.choice, Quot.sound]
```

The formal theorem checks both optimality inequalities and all binary vectors, with only standard Lean axioms. The original development build exposed two pin-specific issues. Lean 4.19.0 Std did not recognize `by_contra` in this file, and `List.length_map` required the mapped function argument. The accepted module uses `Classical.byContradiction` and supplies the argument explicitly.

## Literal odd-field certificate

```sh
python3 research-lab/problems/erdos-117/runs/2026-11-28-graph-covers/check_two_edges_f3.py
```

The test graph has edges `01` and `23` and isolated vertex `4`. The checker uses a product of the four directions in each two-coordinate plane independently of the general support construction. Its clique witnesses have zero on the isolated coordinate. Each covering space spans a pair of those directions and the isolated coordinate line. The checker evaluates the signed determinants on all distinct clique pairs and all pairs within each subspace, and checks union coverage against all $3^5$ vectors. In a separate pass it computes the formula from all $2^5$ supports and explicitly traverses induced-support components.

Observed with Python 3.14.8, exit status zero:

```text
PASS F3: 243 vectors, clique=16, cover=16
PASS support formula: 9 dominating supports, weighted total=16
PASS determinant sign; rejected non-clique and incomplete cover
```

The broken clique changes a point's isolated coordinate to make it orthogonal to a distinct retained point. The broken cover replaces one space by a duplicate of another while keeping the length fixed. A separate determinant control distinguishes subtraction from addition over the odd field.
