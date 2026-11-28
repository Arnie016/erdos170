import Std

set_option autoImplicit false
set_option warningAsError true

namespace OpenMath.GraphCovers

universe u

structure SimpleGraph (Vertex : Type u) where
  edge : Vertex → Vertex → Bool
  symmetric : ∀ i j, edge i j = edge j i
  loopless : ∀ i, edge i i = false

abbrev BinaryVector (Vertex : Type u) := Vertex → Bool

variable {Vertex : Type u}

def orthogonal (graph : SimpleGraph Vertex) (x y : BinaryVector Vertex) : Prop :=
  ∀ i j, graph.edge i j = true → (x i && y j) = (x j && y i)

def dominatesAt (graph : SimpleGraph Vertex) (x : BinaryVector Vertex) (i : Vertex) : Prop :=
  x i = true ∨ ∃ j, x j = true ∧ graph.edge i j = true

def dominates (graph : SimpleGraph Vertex) (x : BinaryVector Vertex) : Prop :=
  ∀ i, dominatesAt graph x i

private theorem orthogonal_self (graph : SimpleGraph Vertex) (x : BinaryVector Vertex) :
    orthogonal graph x x := by
  intro i j _
  exact Bool.and_comm (x i) (x j)

private theorem orthogonal_swap {graph : SimpleGraph Vertex} {x y : BinaryVector Vertex}
    (h : orthogonal graph x y) : orthogonal graph y x := by
  intro i j he
  rw [Bool.and_comm (y i), Bool.and_comm (y j)]
  exact (h i j he).symm

private theorem false_of_not_true {a : Bool} (h : a ≠ true) : a = false := by
  cases a <;> simp_all

private theorem zero_outside_support {graph : SimpleGraph Vertex} {x y : BinaryVector Vertex}
    (hx : dominates graph x) (hxy : orthogonal graph x y) {i : Vertex}
    (hi : x i = false) : y i = false := by
  rcases hx i with hTrue | ⟨j, hj, he⟩
  · simp [hi] at hTrue
  · simpa [hi, hj] using (hxy i j he).symm

/-- Distinct binary vectors of dominating support have nonzero pairing. -/
theorem dominating_orthogonal_eq {graph : SimpleGraph Vertex} {x y : BinaryVector Vertex}
    (hx : dominates graph x) (hy : dominates graph y) (hxy : orthogonal graph x y) :
    x = y := by
  funext i
  cases hxi : x i with
  | false => exact (zero_outside_support hx hxy hxi).symm
  | true =>
    cases hyi : y i with
    | true => rfl
    | false =>
      have hFalse := zero_outside_support hy (orthogonal_swap hxy) hyi
      simp [hxi] at hFalse

private theorem edge_line (a b c d e f : Bool) (hActive : a = true ∨ b = true)
    (hOne : (a && d) = (b && c)) (hTwo : (a && f) = (b && e)) :
    (c && f) = (d && e) := by
  cases a <;> cases b <;> cases c <;> cases d <;> cases e <;> cases f <;> simp_all

/-- The orthogonal space of a vector with dominating support is isotropic. -/
theorem dominating_orthogonal_isotropic {graph : SimpleGraph Vertex}
    {x y z : BinaryVector Vertex} (hx : dominates graph x)
    (hxy : orthogonal graph x y) (hxz : orthogonal graph x z) :
    orthogonal graph y z := by
  intro i j he
  by_cases hActive : x i = true ∨ x j = true
  · exact edge_line (x i) (x j) (y i) (y j) (z i) (z j)
      hActive (hxy i j he) (hxz i j he)
  · have hi := false_of_not_true (fun h : x i = true => hActive (Or.inl h))
    have hj := false_of_not_true (fun h : x j = true => hActive (Or.inr h))
    simp [zero_outside_support hx hxy hi, zero_outside_support hx hxy hj]

def extendsSupport (x y : BinaryVector Vertex) : Prop :=
  ∀ i, x i = true → y i = true

private theorem dominatesAt_mono {graph : SimpleGraph Vertex} {x y : BinaryVector Vertex}
    (hExt : extendsSupport x y) {i : Vertex} (h : dominatesAt graph x i) :
    dominatesAt graph y i := by
  rcases h with hi | ⟨j, hj, he⟩
  · exact Or.inl (hExt i hi)
  · exact Or.inr ⟨j, hExt j hj, he⟩

private def setTrue [DecidableEq Vertex] (x : BinaryVector Vertex) (v : Vertex) :
    BinaryVector Vertex := fun i => if i = v then true else x i

private theorem setTrue_orthogonal [DecidableEq Vertex] {graph : SimpleGraph Vertex}
    {start x : BinaryVector Vertex} {v : Vertex} (hExt : extendsSupport start x)
    (hOrth : orthogonal graph start x) (hMissing : ¬ dominatesAt graph x v) :
    orthogonal graph start (setTrue x v) := by
  have atV : start v = false :=
    false_of_not_true (fun hv => hMissing (Or.inl (hExt v hv)))
  have atNeighbor (j : Vertex) (he : graph.edge v j = true) : start j = false :=
    false_of_not_true (fun hj => hMissing (Or.inr ⟨j, hExt j hj, he⟩))
  intro i j he
  by_cases hi : i = v
  · subst i
    simp [atV, atNeighbor j he]
  by_cases hj : j = v
  · subst j
    have he' : graph.edge v i = true := (graph.symmetric v i).trans he
    simp [atV, atNeighbor i he']
  simpa [setTrue, hi, hj] using hOrth i j he

private theorem extend_over (graph : SimpleGraph Vertex) (start : BinaryVector Vertex)
    (vertices : List Vertex) (current : BinaryVector Vertex)
    (hExt : extendsSupport start current) (hOrth : orthogonal graph start current) :
    ∃ y, extendsSupport current y ∧ orthogonal graph start y ∧
      ∀ v ∈ vertices, dominatesAt graph y v := by
  classical
  induction vertices generalizing current with
  | nil =>
    exact ⟨current, fun _ h => h, hOrth, by simp⟩
  | cons v rest ih =>
    have hStep : ∃ next, extendsSupport current next ∧ orthogonal graph start next ∧
        dominatesAt graph next v := by
      by_cases hSeen : dominatesAt graph current v
      · exact ⟨current, fun _ h => h, hOrth, hSeen⟩
      · have hSet : extendsSupport current (setTrue current v) := by
          intro i hi
          unfold setTrue
          split <;> simp_all
        exact ⟨setTrue current v, hSet, setTrue_orthogonal hExt hOrth hSeen,
          Or.inl (by simp [setTrue])⟩
    obtain ⟨next, hForward, hNext, hAtV⟩ := hStep
    have hNextExt : extendsSupport start next := fun i hi => hForward i (hExt i hi)
    obtain ⟨y, hFinish, hResult, hRest⟩ := ih next hNextExt hNext
    refine ⟨y, (fun i hi => hFinish i (hForward i hi)), hResult, ?_⟩
    intro i hi
    rcases List.mem_cons.mp hi with hEq | hMem
    · subst i
      exact dominatesAt_mono hFinish hAtV
    · exact hRest i hMem

/-- Every vector is orthogonal to a vector with dominating support. A
finite vertex enumeration supplies the bound on the extension procedure. -/
theorem exists_dominating_orthogonal (graph : SimpleGraph Vertex)
    (vertices : List Vertex) (hVertices : ∀ i, i ∈ vertices) (x : BinaryVector Vertex) :
    ∃ y, dominates graph y ∧ orthogonal graph x y := by
  obtain ⟨y, _, hOrth, hDom⟩ :=
    extend_over graph x vertices x (fun _ h => h) (orthogonal_self graph x)
  exact ⟨y, fun i => hDom i (hVertices i), hOrth⟩

private theorem exists_restriction_list (vertices : List Vertex) :
    ∃ vectors : List (BinaryVector Vertex),
      ∀ x : BinaryVector Vertex, ∃ y ∈ vectors, ∀ i ∈ vertices, y i = x i := by
  classical
  induction vertices with
  | nil => exact ⟨[fun _ => false], fun _ => ⟨_, List.mem_cons_self, by simp⟩⟩
  | cons v rest ih =>
    obtain ⟨vectors, hVectors⟩ := ih
    let atV (b : Bool) (y : BinaryVector Vertex) : BinaryVector Vertex :=
      fun i => if i = v then b else y i
    refine ⟨vectors.flatMap (fun y => [atV false y, atV true y]), ?_⟩
    intro x
    obtain ⟨y, hy, hAgree⟩ := hVectors x
    refine ⟨atV (x v) y, List.mem_flatMap.mpr ⟨y, hy, ?_⟩, ?_⟩
    · cases x v <;> simp
    · intro i hi
      by_cases hEq : i = v
      · subst i
        simp [atV]
      · have hRest := (List.mem_cons.mp hi).resolve_left hEq
        simp [atV, hEq, hAgree i hRest]

private theorem exists_distinct_filter {Item : Type u} (p : Item → Prop) (items : List Item) :
    ∃ filtered : List Item, filtered.Nodup ∧
      ∀ x, x ∈ filtered ↔ x ∈ items ∧ p x := by
  classical
  induction items with
  | nil => exact ⟨[], List.nodup_nil, by simp⟩
  | cons head tail ih =>
    obtain ⟨filtered, hDistinct, hMem⟩ := ih
    by_cases hAdd : p head ∧ head ∉ filtered
    · refine ⟨head :: filtered, List.nodup_cons.mpr ⟨hAdd.2, hDistinct⟩, ?_⟩
      intro x
      simp only [List.mem_cons, hMem]
      constructor
      · rintro (rfl | h)
        · exact ⟨Or.inl rfl, hAdd.1⟩
        · exact ⟨Or.inr h.1, h.2⟩
      · rintro ⟨rfl | h, hp⟩
        · exact Or.inl rfl
        · exact Or.inr ⟨h, hp⟩
    · refine ⟨filtered, hDistinct, ?_⟩
      intro x
      constructor
      · intro hx
        obtain ⟨ht, hp⟩ := (hMem x).mp hx
        exact ⟨List.mem_cons_of_mem head ht, hp⟩
      · rintro ⟨hx, hp⟩
        rcases List.mem_cons.mp hx with rfl | ht
        · exact Classical.byContradiction (fun hn => hAdd ⟨hp, hn⟩)
        · exact (hMem x).mpr ⟨ht, hp⟩

private theorem exists_dominating_support_list (graph : SimpleGraph Vertex)
    (vertices : List Vertex) (hVertices : ∀ i, i ∈ vertices) :
    ∃ supports : List (BinaryVector Vertex), supports.Nodup ∧
      ∀ x, x ∈ supports ↔ dominates graph x := by
  obtain ⟨vectors, hRestriction⟩ := exists_restriction_list vertices
  have hAll (x : BinaryVector Vertex) : x ∈ vectors := by
    obtain ⟨y, hy, hAgree⟩ := hRestriction x
    have hEq : y = x := funext (fun i => hAgree i (hVertices i))
    simpa [hEq] using hy
  obtain ⟨supports, hDistinct, hMem⟩ := exists_distinct_filter (dominates graph) vectors
  exact ⟨supports, hDistinct, fun x => (hMem x).trans (and_iff_right (hAll x))⟩

/-- Over the two-element field, addition is XOR and scalar action is AND. -/
def IsIsotropicSubspace (graph : SimpleGraph Vertex) (space : BinaryVector Vertex → Prop) :
    Prop :=
  space (fun _ => false) ∧
  (∀ x y, space x → space y → space (fun i => xor (x i) (y i))) ∧
  (∀ c : Bool, ∀ x, space x → space (fun i => c && x i)) ∧
  (∀ x y, space x → space y → orthogonal graph x y)

private theorem edge_line_add (a b c d e f : Bool)
    (hOne : (a && d) = (b && c)) (hTwo : (a && f) = (b && e)) :
    (a && xor d f) = (b && xor c e) := by
  cases a <;> cases b <;> cases c <;> cases d <;> cases e <;> cases f <;> simp_all

private theorem centralizer_isotropic_subspace (graph : SimpleGraph Vertex)
    (x : BinaryVector Vertex) (hx : dominates graph x) :
    IsIsotropicSubspace graph (orthogonal graph x) := by
  refine ⟨?_, ?_, ?_, fun _ _ hy hz => dominating_orthogonal_isotropic hx hy hz⟩
  · intro i j _
    simp
  · intro y z hy hz i j he
    exact edge_line_add (x i) (x j) (y i) (y j) (z i) (z j) (hy i j he) (hz i j he)
  · intro c y hy i j he
    cases c
    · simp
    · simpa using hy i j he

def IsClique (graph : SimpleGraph Vertex) (points : List (BinaryVector Vertex)) : Prop :=
  points.Pairwise (fun x y => ¬ orthogonal graph x y)

def IsCover (graph : SimpleGraph Vertex)
    (spaces : List (BinaryVector Vertex → Prop)) : Prop :=
  (∀ space ∈ spaces, IsIsotropicSubspace graph space) ∧
  (∀ x, ∃ space ∈ spaces, space x)

private theorem dominating_list_clique (graph : SimpleGraph Vertex)
    (points : List (BinaryVector Vertex)) (hDistinct : points.Nodup)
    (hDom : ∀ x ∈ points, dominates graph x) : IsClique graph points := by
  induction points with
  | nil => exact List.Pairwise.nil
  | cons x tail ih =>
    obtain ⟨hNotMem, hDistinctTail⟩ := List.nodup_cons.mp hDistinct
    apply List.pairwise_cons.mpr
    have hTail (y : BinaryVector Vertex) (hy : y ∈ tail) : dominates graph y :=
      hDom y (List.mem_cons_of_mem x hy)
    refine ⟨?_, ih hDistinctTail hTail⟩
    intro y hy hOrth
    have hEq := dominating_orthogonal_eq (hDom x List.mem_cons_self) (hTail y hy) hOrth
    exact hNotMem (hEq.symm ▸ hy)

private theorem clique_le_cover_of_points (graph : SimpleGraph Vertex)
    (points : List (BinaryVector Vertex)) (spaces : List (BinaryVector Vertex → Prop))
    (hClique : IsClique graph points)
    (hIsotropic : ∀ space ∈ spaces, ∀ x y, space x → space y → orthogonal graph x y)
    (hCovered : ∀ x ∈ points, ∃ space ∈ spaces, space x) :
    points.length ≤ spaces.length := by
  classical
  induction points generalizing spaces with
  | nil => simp
  | cons x tail ih =>
    obtain ⟨hHead, hTail⟩ := List.pairwise_cons.mp hClique
    obtain ⟨space, hMem, hInside⟩ := hCovered x List.mem_cons_self
    have hCoveredTail : ∀ y ∈ tail, ∃ other ∈ spaces.erase space, other y := by
      intro y hy
      obtain ⟨other, hOther, hyInside⟩ := hCovered y (List.mem_cons_of_mem x hy)
      have hNe : other ≠ space := by
        intro hEq
        subst other
        exact hHead y hy (hIsotropic space hMem x y hInside hyInside)
      exact ⟨other, (List.mem_erase_of_ne hNe).mpr hOther, hyInside⟩
    have hBound := ih (spaces.erase space) hTail
      (fun other hOther => hIsotropic other (List.mem_of_mem_erase hOther)) hCoveredTail
    rw [List.length_erase_of_mem hMem] at hBound
    have hPositive := List.length_pos_of_mem hMem
    simp only [List.length_cons]
    omega

theorem clique_le_cover (graph : SimpleGraph Vertex) (points : List (BinaryVector Vertex))
    (spaces : List (BinaryVector Vertex → Prop)) (hClique : IsClique graph points)
    (hCover : IsCover graph spaces) : points.length ≤ spaces.length :=
  clique_le_cover_of_points graph points spaces hClique
    (fun space hm => (hCover.1 space hm).2.2.2) (fun x _ => hCover.2 x)

/-- The duplicate-free enumeration of all dominating binary supports is a
maximum clique. Its orthogonal spaces give an isotropic cover of equal
length, and every isotropic cover has at least that length. -/
theorem binary_clique_cover_domination (graph : SimpleGraph Vertex)
    (vertices : List Vertex) (hVertices : ∀ i, i ∈ vertices) :
    ∃ supports : List (BinaryVector Vertex), supports.Nodup ∧
      (∀ x, x ∈ supports ↔ dominates graph x) ∧
      IsClique graph supports ∧
      (∀ points, IsClique graph points → points.length ≤ supports.length) ∧
      ∃ spaces : List (BinaryVector Vertex → Prop),
        IsCover graph spaces ∧ spaces.length = supports.length ∧
        (∀ cover, IsCover graph cover → supports.length ≤ cover.length) := by
  obtain ⟨supports, hDistinct, hMem⟩ := exists_dominating_support_list graph vertices hVertices
  have hClique := dominating_list_clique graph supports hDistinct (fun x hx => (hMem x).mp hx)
  let spaces := supports.map (orthogonal graph)
  have hCover : IsCover graph spaces := by
    constructor
    · intro space hSpace
      obtain ⟨x, hx, hEq⟩ := List.mem_map.mp hSpace
      exact hEq ▸ centralizer_isotropic_subspace graph x ((hMem x).mp hx)
    · intro x
      obtain ⟨y, hy, hOrth⟩ := exists_dominating_orthogonal graph vertices hVertices x
      exact ⟨orthogonal graph y, List.mem_map.mpr ⟨y, (hMem y).mpr hy, rfl⟩,
        orthogonal_swap hOrth⟩
  have hLength : spaces.length = supports.length := List.length_map _
  refine ⟨supports, hDistinct, hMem, hClique, ?_, spaces, hCover, hLength, ?_⟩
  · intro points hPoints
    exact hLength ▸ clique_le_cover graph points spaces hPoints hCover
  · intro cover hAny
    exact clique_le_cover graph supports cover hClique hAny

#print axioms dominating_orthogonal_eq
#print axioms dominating_orthogonal_isotropic
#print axioms exists_dominating_orthogonal
#print axioms binary_clique_cover_domination

end OpenMath.GraphCovers
