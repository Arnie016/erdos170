# Lean formalization

Toolchain: Lean 4.19.0, using only Std. Run `lake build` here.

## November 28 additions

The [binary graphical theorem](OpenMath/GraphCovers.lean) constructs a maximum clique and matching minimum subspace cover for an arbitrary finite simple graph. The finite-field extension and group lifting have written proofs. [Group proof and scope](../problems/erdos-117/runs/2026-11-28-graph-covers/PROOF.md)

For sparse rulers, the pin checks a generic one-mark counting obstruction, the statement translation, residue capacity, modular separation and an explicit rational error bound assuming the two geometric inequalities. The ordinary argument proves the geometric inequalities. [Residue lemmas](OpenMath/ExteriorRepair.lean) · [Deficit lemmas](OpenMath/DeficitBounds.lean) · [Geometric proof](../problems/erdos-170/runs/2026-11-28-exterior-repair/DEFICIT-BOUNDS.md)

Run `lake build --wfail` and then `python3 -B audit_axioms.py` in this directory. The audit checks the pinned version, the frozen mixed statement, each printed dependency and the presence of named target theorems.

## Initial October witnesses

The initial 12 declarations are **LEAN_VERIFIED_AT_COMMIT e4c0bc8ac647538f5dd22c84613b6745446f7bfe**, via [GitHub Actions run 37905339322](https://github.com/Arnie016/erdos170/actions/runs/37905339322).

`OpenMath/Certificates.lean` checks distinct finite clique lists, sizes, pairwise noncommutation, coverage, XOR closure and simultaneous isotropy for two order-64 commutator examples. It also proves product noncommutation from one coordinate. The graph predicates use binary coordinates encoded by Nat values 0 through 15.

The axiom audit reports no dependencies for ten declarations. `coverA_checked` and `coverB_checked` depend only on Lean's standard `propext`; none depend on `sorryAx` or native-evaluation/compiler-trust axioms. The CI workflow prints and preserves this report.

**Not yet formalized:** the abstract group-to-graph reduction, exact cover-number optimality as an abstract theorem, the 2,825-kernel census, the 174,251-pencil census and the all-n modular conjecture. Successful compilation of these witnesses is not a blanket formalization of the repository.

The first workflow failed on a missing Lake manifest before checking proofs. The manifest is now committed. The October publication container used the GitHub runner for formal verification. The November additions also built locally on the pinned compiler.
