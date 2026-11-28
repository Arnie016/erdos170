import OpenMath.MixedRepairTarget

set_option autoImplicit false
set_option warningAsError true

namespace OpenMath.SparseRuler

private theorem length_le_of_subset_of_nodup (small large : List Int)
    (hDistinct : small.Nodup) (hSubset : small ⊆ large) :
    small.length ≤ large.length := by
  induction small generalizing large with
  | nil => simp
  | cons x xs ih =>
    obtain ⟨hNotMem, hDistinctTail⟩ := List.nodup_cons.mp hDistinct
    have hx : x ∈ large := hSubset List.mem_cons_self
    have hTail : xs ⊆ large.erase x := by
      intro y hy
      have hNe : y ≠ x := by
        intro hEq
        subst y
        exact hNotMem hy
      exact (List.mem_erase_of_ne hNe).mpr (hSubset (List.mem_cons_of_mem x hy))
    have hBound := ih (large.erase x) hDistinctTail hTail
    rw [List.length_erase_of_mem hx] at hBound
    have hPositive := List.length_pos_of_mem hx
    simp only [List.length_cons]
    omega

def markDistance (z x : Int) : Int := if x < z then z - x else x - z

theorem new_length_is_incident (old : List Int) (z d : Int)
    (hNew : positiveDifference (fun x => x = z ∨ x ∈ old) d)
    (hMissing : ¬ positiveDifference (fun x => x ∈ old) d) :
    d ∈ old.map (markDistance z) := by
  obtain ⟨x, y, hx, hy, hxy, hLength⟩ := hNew
  rcases hx with hx | hx <;> rcases hy with hy | hy
  · omega
  · subst x
    apply List.mem_map.mpr
    exact ⟨y, hy, by rw [markDistance, if_neg (by omega), hLength]⟩
  · subst y
    apply List.mem_map.mpr
    exact ⟨x, hx, by rw [markDistance, if_pos hxy, hLength]⟩
  · exact (hMissing ⟨x, y, hx, hy, hxy, hLength⟩).elim

/-- Any duplicate-free list of newly covered lengths has size at most the
length of the surviving marks list. A duplicate-free marks list gives the
number of surviving marks. -/
theorem one_new_mark_count (old : List Int) (z : Int) (gains : List Int)
    (hDistinct : gains.Nodup)
    (hGains : ∀ d ∈ gains,
      positiveDifference (fun x => x = z ∨ x ∈ old) d ∧
        ¬ positiveDifference (fun x => x ∈ old) d) :
    gains.length ≤ old.length := by
  have hSubset : gains ⊆ old.map (markDistance z) := by
    intro d hd
    exact new_length_is_incident old z d (hGains d hd).1 (hGains d hd).2
  simpa using length_le_of_subset_of_nodup gains (old.map (markDistance z)) hDistinct hSubset

/-- Conditional obstruction once a duplicate-free list of too many missing
target lengths has been exhibited. -/
theorem too_many_missing_for_one_mark (old : List Int) (z n : Int) (lost : List Int)
    (hDistinct : lost.Nodup)
    (hLost : ∀ d ∈ lost, 1 ≤ d ∧ d ≤ n ∧
      ¬ positiveDifference (fun x => x ∈ old) d)
    (hCount : old.length < lost.length) :
    ¬ covers (fun x => x = z ∨ x ∈ old) n := by
  intro hCovers
  apply (Nat.not_le_of_gt hCount)
  apply one_new_mark_count old z lost hDistinct
  intro d hd
  obtain ⟨hLow, hHigh, hMissing⟩ := hLost d hd
  exact ⟨(hCovers d).mpr ⟨hLow, hHigh⟩, hMissing⟩

#print axioms new_length_is_incident
#print axioms one_new_mark_count
#print axioms too_many_missing_for_one_mark

end OpenMath.SparseRuler
