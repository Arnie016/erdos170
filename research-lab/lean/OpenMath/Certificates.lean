import Std

/-
Explicit finite witnesses from the research notebook.
The full abstract group-to-graph reduction is not formalized in this file.
Bit i is the i-th binary coordinate; the domain is the integers 0,...,15.
-/
namespace OpenMath

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

def bit (v i : Nat) : Nat := (v / 2^i) % 2

def wedge (i j v w : Nat) : Nat :=
  (bit v i * bit w j + bit v j * bit w i) % 2

def noncommA (v w : Nat) : Bool :=
  (wedge 0 1 v w != 0) || (wedge 2 3 v w != 0)

def noncommB (v w : Nat) : Bool :=
  (wedge 0 1 v w != 0) || (wedge 1 2 v w != 0)

def cliqueA : List Nat := [5, 9, 13, 6, 10, 14, 7, 11, 15]
def cliqueB : List Nat := [1, 2, 3, 6, 7]

def coverA : List (List Nat) :=
  [[0,1,4,5], [0,1,8,9], [0,1,12,13],
   [0,2,4,6], [0,2,8,10], [0,2,12,14],
   [0,3,4,7], [0,3,8,11], [0,3,12,15]]

def coverB : List (List Nat) :=
  [[0,1,4,5,8,9,12,13], [0,2,8,10], [0,3,8,11],
   [0,6,8,14], [0,7,8,15]]

def cliqueCheck (nc : Nat -> Nat -> Bool) (xs : List Nat) : Bool :=
  xs.all fun v => xs.all fun w => (v == w) || nc v w

def coverCheck (nc : Nat -> Nat -> Bool) (ss : List (List Nat)) : Bool :=
  ((List.range 16).all fun v => ss.any fun s => s.contains v) &&
  (ss.all fun s => s.contains 0 &&
    (s.all fun v => v < 16) &&
    (s.all fun v => s.all fun w =>
      s.contains (Nat.xor v w) && !(nc v w)))

theorem cliqueA_distinct : cliqueA.Nodup := by decide
theorem cliqueB_distinct : cliqueB.Nodup := by decide
theorem cliqueA_size : cliqueA.length = 9 := by decide
theorem cliqueB_size : cliqueB.length = 5 := by decide
theorem cliqueA_checked : cliqueCheck noncommA cliqueA = true := by decide
theorem cliqueB_checked : cliqueCheck noncommB cliqueB = true := by decide
theorem coverA_size : coverA.length = 9 := by decide
theorem coverB_size : coverB.length = 5 := by decide
theorem coverA_checked : coverCheck noncommA coverA = true := by decide
theorem coverB_checked : coverCheck noncommB coverB = true := by decide

def productOp {G H : Type} (g : G -> G -> G) (h : H -> H -> H)
    (p q : G × H) : G × H := (g p.1 q.1, h p.2 q.2)

theorem product_noncommutes_left {G H : Type}
    (g : G -> G -> G) (h : H -> H -> H)
    (x y : G) (u v : H) (hn : g x y ≠ g y x) :
    productOp g h (x,u) (y,v) ≠ productOp g h (y,v) (x,u) := by
  intro he
  exact hn (congrArg Prod.fst he)

theorem product_noncommutes_right {G H : Type}
    (g : G -> G -> G) (h : H -> H -> H)
    (x y : G) (u v : H) (hn : h u v ≠ h v u) :
    productOp g h (x,u) (y,v) ≠ productOp g h (y,v) (x,u) := by
  intro he
  exact hn (congrArg Prod.snd he)

#print axioms cliqueA_distinct
#print axioms cliqueB_distinct
#print axioms cliqueA_size
#print axioms cliqueB_size
#print axioms cliqueA_checked
#print axioms cliqueB_checked
#print axioms coverA_size
#print axioms coverB_size
#print axioms coverA_checked
#print axioms coverB_checked
#print axioms product_noncommutes_left
#print axioms product_noncommutes_right

end OpenMath
