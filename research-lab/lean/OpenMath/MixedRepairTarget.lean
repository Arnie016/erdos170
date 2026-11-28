import Std

set_option autoImplicit false

namespace OpenMath.SparseRuler

def positiveDifference (marks : Int → Prop) (d : Int) : Prop :=
  ∃ x y : Int, marks x ∧ marks y ∧ x < y ∧ y - x = d

def covers (marks : Int → Prop) (n : Int) : Prop :=
  ∀ d : Int, positiveDifference marks d ↔ 1 ≤ d ∧ d ≤ n

def stepA (r : Int) : Int := 2 * r + 1
def stepB (r : Int) : Int := 2 * r + 2
def stepQ (r : Int) : Int := stepA r + stepB r
def middleCount (r : Int) : Int := 2 * r
def middleStart (r : Int) : Int := (r + 1) * stepA r
def rulerLength (r : Int) : Int :=
  middleStart r + middleCount r * stepQ r + r + (r + 1) * stepB r

def block0 (r x : Int) : Prop := 0 ≤ x ∧ x ≤ r

def block1 (r x : Int) : Prop :=
  ∃ i : Int, 1 ≤ i ∧ i ≤ r ∧ x = i * stepA r

def middleMark (r j : Int) : Int := middleStart r + j * stepQ r

def block2 (r x : Int) : Prop :=
  ∃ j : Int, 0 ≤ j ∧ j ≤ middleCount r ∧ x = middleMark r j

def block3 (r x : Int) : Prop :=
  ∃ i : Int, 1 ≤ i ∧ i ≤ r ∧ x = rulerLength r - r - i * stepB r

def block4 (r x : Int) : Prop := rulerLength r - r ≤ x ∧ x ≤ rulerLength r

def exterior (r x : Int) : Prop := block0 r x ∨ block1 r x ∨ block3 r x ∨ block4 r x

def wichmann (r x : Int) : Prop := block2 r x ∨ exterior r x

def survivors (r j v x : Int) : Prop := wichmann r x ∧ x ≠ middleMark r j ∧ x ≠ v

def mixedRepair (r j v z x : Int) : Prop := x = z ∨ survivors r j v x

/-- Literal existential question from the frozen statement. -/
def mixedRepairExists : Prop :=
  ∃ r j v z : Int,
    1 ≤ r ∧
    1 ≤ j ∧ j ≤ middleCount r - 1 ∧
    exterior r v ∧
    0 ≤ z ∧ z ≤ rulerLength r ∧ ¬ wichmann r z ∧
    covers (mixedRepair r j v z) (rulerLength r)

def mixedRepairObstruction : Prop := ¬ mixedRepairExists

end OpenMath.SparseRuler
