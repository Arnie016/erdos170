import OpenMath.MixedRepairTarget

set_option autoImplicit false
set_option warningAsError true

namespace OpenMath.SparseRuler

/-- A strict-middle-to-left-progression distance divisible by the right
progression step exceeds the small multiples on that step. -/
theorem left_ap_distance_gt_of_stepB_dvd (r i j : Int) (hr : 1 ≤ r)
    (hi : 1 ≤ i ∧ i ≤ r) (hj : 1 ≤ j ∧ j ≤ 2 * r - 1)
    (hDiv : stepB r ∣ (r + 1 - i) * stepA r + j * stepQ r) :
    2 * (r - 1) * stepB r < (r + 1 - i) * stepA r + j * stepQ r := by
  let a := stepA r
  let b := stepB r
  let m := r + 1 - i
  change b ∣ m * a + j * (a + b) at hDiv
  change 2 * (r - 1) * b < m * a + j * (a + b)
  have ha : 0 ≤ a := by
    dsimp [a, stepA]
    omega
  have hb : 0 < b := by
    dsimp [b, stepB]
    omega
  have hAdjacent : a = b - 1 := by
    dsimp [a, b, stepA, stepB]
    omega
  have hExpand : m * a + j * (a + b) = (m + j) * a + j * b := by
    simp only [Int.mul_add, Int.add_mul, Int.add_assoc]
  have hResidue : m * a + j * (a + b) = (m + j + j) * b - (m + j) := by
    rw [hExpand, hAdjacent]
    simp only [Int.mul_sub, Int.mul_one, Int.add_mul]
    omega
  have hResidueDvd : b ∣ (m + j + j) * b - (m + j) := hResidue ▸ hDiv
  have hSumDvd : b ∣ m + j :=
    (Int.dvd_iff_dvd_of_dvd_sub hResidueDvd).mp (Int.dvd_mul_left _ _)
  have hSize : b ≤ m + j := Int.le_of_dvd (by dsimp [m]; omega) hSumDvd
  have hGap : 2 * (r - 1) < a := by
    dsimp [a, stepA]
    omega
  calc
    2 * (r - 1) * b < a * b := Int.mul_lt_mul_of_pos_right hGap hb
    _ = b * a := Int.mul_comm _ _
    _ ≤ (m + j) * a := Int.mul_le_mul_of_nonneg_right hSize ha
    _ ≤ (m + j) * a + j * b :=
      Int.le_add_of_nonneg_right (Int.mul_nonneg (by omega) (Int.le_of_lt hb))
    _ = m * a + j * (a + b) := hExpand.symm

/-- A strict-middle-to-right-progression distance divisible by the left
progression step exceeds the small multiples on that step. The index is
measured from the right junction. -/
theorem right_ap_distance_gt_of_stepA_dvd (r j k : Int) (hr : 1 ≤ r)
    (hj : 1 ≤ j ∧ j ≤ 2 * r - 1) (hk : 1 ≤ k ∧ k ≤ r)
    (hDiv : stepA r ∣ (2 * r - j) * stepQ r + k * stepB r) :
    2 * (r - 1) * stepA r < (2 * r - j) * stepQ r + k * stepB r := by
  let a := stepA r
  let b := stepB r
  let m := 2 * r - j
  change a ∣ m * (a + b) + k * b at hDiv
  change 2 * (r - 1) * a < m * (a + b) + k * b
  have ha : 0 < a := by
    dsimp [a, stepA]
    omega
  have hb : 0 ≤ b := by
    dsimp [b, stepB]
    omega
  have hAdjacent : b = a + 1 := by
    dsimp [a, b, stepA, stepB]
    omega
  have hExpand : m * (a + b) + k * b = m * a + (m + k) * b := by
    simp only [Int.mul_add, Int.add_mul, Int.add_assoc]
  have hResidue : m * (a + b) + k * b = (m + (m + k)) * a + (m + k) := by
    rw [hExpand, hAdjacent]
    simp only [Int.mul_add, Int.mul_one, Int.add_mul, Int.add_assoc]
  have hSumDvd : a ∣ m + k := by simpa only [hResidue, Int.dvd_mul_self_add] using hDiv
  have hSize : a ≤ m + k := Int.le_of_dvd (by dsimp [m]; omega) hSumDvd
  have hGap : 2 * (r - 1) < b := by
    dsimp [b, stepB]
    omega
  calc
    2 * (r - 1) * a < b * a := Int.mul_lt_mul_of_pos_right hGap ha
    _ = a * b := Int.mul_comm _ _
    _ ≤ (m + k) * b := Int.mul_le_mul_of_nonneg_right hSize hb
    _ ≤ m * a + (m + k) * b :=
      Int.le_add_of_nonneg_left (Int.mul_nonneg (by dsimp [m]; omega) (Int.le_of_lt ha))
    _ = m * (a + b) + k * b := hExpand.symm

/-- Clear denominators in the positive rational error bound with tolerance
numerator / denominator. The geometric deficit bounds are hypotheses. -/
theorem normalized_deficit_error_lt (r deficit numerator denominator : Int)
    (hPositive : 0 < numerator ∧ 0 < denominator)
    (hLower : r - 8 ≤ deficit) (hUpper : deficit ≤ r + 5)
    (hThreshold : 8 * denominator < numerator * r) :
    denominator * ((deficit - r).natAbs : Int) < numerator * r := by
  have hAbs : ((deficit - r).natAbs : Int) ≤ 8 := by
    by_cases hNonneg : 0 ≤ deficit - r
    · rw [Int.natAbs_of_nonneg hNonneg]
      omega
    · rw [← Int.natAbs_neg (deficit - r), Int.natAbs_of_nonneg (by omega)]
      omega
  have hScaled : denominator * ((deficit - r).natAbs : Int) ≤ denominator * 8 :=
    Int.mul_le_mul_of_nonneg_left hAbs (Int.le_of_lt hPositive.2)
  exact Int.lt_of_le_of_lt hScaled (by simpa [Int.mul_comm] using hThreshold)

/-- An explicit eventual threshold for each positive rational tolerance. -/
theorem normalized_deficit_error_eventually_lt (deficit : Int → Int)
    (hBounds : ∀ r, 2 ≤ r → r - 8 ≤ deficit r ∧ deficit r ≤ r + 5)
    (numerator denominator : Int) (hPositive : 0 < numerator ∧ 0 < denominator) :
    ∀ r, 8 * denominator + 1 ≤ r →
      denominator * ((deficit r - r).natAbs : Int) < numerator * r := by
  intro r hr
  have hOne : 1 ≤ numerator := by omega
  have hR : r ≤ numerator * r := by
    simpa using Int.mul_le_mul_of_nonneg_right hOne (show 0 ≤ r by omega)
  obtain ⟨hLower, hUpper⟩ := hBounds r (by omega)
  exact normalized_deficit_error_lt r (deficit r) numerator denominator hPositive
    hLower hUpper (Int.lt_of_lt_of_le (by omega) hR)

#print axioms left_ap_distance_gt_of_stepB_dvd
#print axioms right_ap_distance_gt_of_stepA_dvd
#print axioms normalized_deficit_error_lt
#print axioms normalized_deficit_error_eventually_lt

end OpenMath.SparseRuler
