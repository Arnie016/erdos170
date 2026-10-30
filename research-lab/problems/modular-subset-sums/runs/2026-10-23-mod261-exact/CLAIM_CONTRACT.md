# Frozen contract | 23 October2026

Decide the complete n8,N261 class under the exact independent-sign/unit-dilation convention inherited from n7. Every labelled subset sum, including zero, must be distinct. Falsifier: an admissible set outside the three published perturbation classes, or with maximum dyadic overlap<=6. Do not hardcode three classes into search acceptance.

Proved normalization: every gcd(a,261)<=5; at most one member is divisible by3. Thus a unit exists and can be scaled to1. Search all increasing eight-subsets of1,...,130 containing1 under these necessary conditions, pruning only collisions. Worker A maintains subset sums; worker B maintains differences and directly checks every leaf's256 sums. Both sources frozen before worker1.

At most two sequential workers, each35seconds including compilation, with kill/wait cleanup. No RNG. Accept a complete certificate only when both full accepted lists agree, orbit expansion and dyadic overlaps are checked, and negative controls fail as expected. A timeout yields no complete theorem. Record source hashes,commands,raw leaves,counts and failures. This closes a named finite uncertainty, not an unbounded theorem.
