import OpenMath.MixedRepairTarget

set_option autoImplicit false

namespace OpenMath.SparseRuler

/-- Compare the definition-only target to an inlined translation of
STATEMENT.md without loading any of the proof code. -/
example : mixedRepairExists =
    (∃ r j v z : Int,
      let s := 2 * r
      let a := 2 * r + 1
      let b := 2 * r + 2
      let q := a + b
      let l := (r + 1) * a
      let n := l + s * q + r + (r + 1) * b
      let outside : Int → Prop := fun t =>
        (0 ≤ t ∧ t ≤ r) ∨
        (∃ i : Int, 1 ≤ i ∧ i ≤ r ∧ t = i * a) ∨
        (∃ i : Int, 1 ≤ i ∧ i ≤ r ∧ t = n - r - i * b) ∨
        (n - r ≤ t ∧ t ≤ n)
      let original : Int → Prop := fun t =>
        (∃ i : Int, 0 ≤ i ∧ i ≤ s ∧ t = l + i * q) ∨ outside t
      let repaired : Int → Prop := fun t =>
        t = z ∨ (original t ∧ t ≠ l + j * q ∧ t ≠ v)
      1 ≤ r ∧ 1 ≤ j ∧ j ≤ s - 1 ∧ outside v ∧
      0 ≤ z ∧ z ≤ n ∧ ¬ original z ∧
      ∀ d : Int, (∃ x y : Int, repaired x ∧ repaired y ∧ x < y ∧ y - x = d) ↔
        1 ≤ d ∧ d ≤ n) := rfl

end OpenMath.SparseRuler
