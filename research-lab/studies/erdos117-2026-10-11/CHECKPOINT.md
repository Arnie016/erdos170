# CHECKPOINT (2026-10-11)

Active #117 finite question: three-coordinate alternating maps on F2^5. A 35-second-bounded full orbit census of all 6,347,715 three-planes produced 22 GL(5,2) orbits, with chi=omega on every tested orbit representative. The computation is one complete exact implementation; an independent full checker and explicit vertex witnesses are pending. The all-group bound h(n) is unchanged. No #170 work occurred.

Dependencies: published 2026-10-08 dimension<=4 census, 2026-10-09 dimension5 derived-rank<=2 census, elementary bilinear group reduction. Do not import historical recurrence claims. For the new 2026-10-11 code and receipt see `src/orbits.cpp`, `results/worker2.log`, `proofs/rank5_three_coordinate_orbits.md`.

NEXT: Build an independent exact checker for 22 orbit representatives, reconstruct actual colored independent sets and maximum cliques, then verify the full orbit-size count by a distinct orbit-generation path. A mismatch falsifies the class claim. Parked: Lean group-cover correspondence. Nothing in this package proves the optimal uniform constant.
