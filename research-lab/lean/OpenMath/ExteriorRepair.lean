import Std

namespace OpenMath.ExteriorRepair

/-- A list of successful partners, classified by their residues. Two allowed
residues give at most two partners when residues distinguish all partners. -/
theorem two_residue_capacity {α : Type} (residue : α → Int)
    (partners : List α) (u v : Int)
    (unique : (partners.map residue).Nodup)
    (allowed : ∀ x ∈ partners, residue x = u ∨ residue x = v) :
    partners.length ≤ 2 := by
  cases partners with
  | nil => simp
  | cons x tail =>
    cases tail with
    | nil => simp
    | cons y rest =>
      cases rest with
      | nil => simp
      | cons z more =>
        have hx := allowed x (by simp)
        have hy := allowed y (by simp)
        have hz := allowed z (by simp)
        simp [List.Nodup] at unique
        rcases hx with hx | hx <;> rcases hy with hy | hy <;> rcases hz with hz | hz <;> simp_all

/-- Absolute differences supply two possible residue classes. The first
argument is the fresh integer and the second is a possible partner. -/
theorem signed_partner_residues (z p d m : Int)
    (hit : (if z ≤ p then p - z else z - p) % m = d % m) :
    p % m = (z + d) % m ∨ p % m = (z - d) % m := by
  split at hit
  · left
    calc
      p % m = (z + (p - z)) % m := by congr 1; omega
      _ = (z + d) % m := by rw [Int.add_emod z (p-z), hit, ← Int.add_emod]
  · right
    calc
      p % m = (z - (z - p)) % m := by congr 1; omega
      _ = (z - d) % m := by rw [Int.sub_emod z (z-p), hit, ← Int.sub_emod]

/-- This is the one-block estimate used by both the exterior column proof and
the extension's progression-localization argument. -/
theorem one_block_capacity (partners : List Int) (z d m : Int)
    (unique : (partners.map (fun p => p % m)).Nodup)
    (hits : ∀ p ∈ partners, (if z ≤ p then p-z else z-p) % m = d % m) :
    partners.length ≤ 2 := by
  apply two_residue_capacity (fun p => p % m) partners ((z+d)%m) ((z-d)%m) unique
  intro p hp
  exact signed_partner_residues z p d m (hits p hp)

/-- The forbidden cap congruence after localization at a deleted junction. -/
theorem cap_congruence_impossible (r x k : Int)
    (hr : 17 ≤ r) (hlo : 0 ≤ x) (hhi : x ≤ r) (kl : -8 ≤ k) (kh : k ≤ 8) :
    (2*x + r + 1 + k) % (4*r + 3) ≠ 0 := by
  rw [Int.emod_eq_of_lt (by omega) (by omega)]
  omega

#print axioms two_residue_capacity
#print axioms signed_partner_residues
#print axioms one_block_capacity
#print axioms cap_congruence_impossible

end OpenMath.ExteriorRepair
