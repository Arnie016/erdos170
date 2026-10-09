# Lean formalization

Toolchain: Lean 4.19.0, using only Std. Run `lake build` here.

The initial 12 declarations are **LEAN_VERIFIED_AT_COMMIT e4c0bc8ac647538f5dd22c84613b6745446f7bfe**, via [GitHub Actions run 37905339322](https://github.com/Arnie016/erdos170/actions/runs/37905339322).

`OpenMath/Certificates.lean` checks distinct finite clique lists, sizes, pairwise noncommutation, coverage, XOR closure and simultaneous isotropy for two order-64 commutator examples. It also proves product noncommutation from one coordinate. The graph predicates use binary coordinates encoded by Nat values 0 through 15.

The axiom audit reports no dependencies for ten declarations. `coverA_checked` and `coverB_checked` depend only on Lean's standard `propext`; none depend on `sorryAx` or native-evaluation/compiler-trust axioms. The CI workflow prints and preserves this report.

**Not yet formalized:** the abstract group-to-graph reduction, exact cover-number optimality as an abstract theorem, the 2,825-kernel census, the 174,251-pencil census and the all-n modular conjecture. Successful compilation of these witnesses is not a blanket formalization of the repository.

The first workflow failed on a missing Lake manifest before checking proofs. The manifest is now committed. Lean was not installed in the local publication container; the actual formal verification occurred on the GitHub runner.
