# Lean formalization

Pinned version: Lean 4.19.0. Run `lake build` from this directory.

Initial state: LEAN_PENDING until CI for the exact uploaded commit is observed to pass. Lean was not installed in the publication container; direct toolchain download was unavailable there.

`OpenMath/Certificates.lean` states explicit finite witnesses for the two order-64 commutator examples, using Nat bit coordinates for F2^4. It checks distinct clique lists, coverage, XOR closure and simultaneous isotropy. It also proves an elementary product-noncommutation implication without imposing any unproved group axioms.

The historical large finite censuses are NOT yet Lean proofs. A successful build of this small library does not change that. No unfinished theorem is represented by an admitted proof.
