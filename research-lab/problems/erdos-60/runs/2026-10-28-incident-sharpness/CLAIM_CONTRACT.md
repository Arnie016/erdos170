# Frozen claim contract
Question: For every odd prime power q>=3, orthogonal polarity H of PG(2,q), two incident added nonedges and one removed H-edge, is the C4 count >=2q-3 and is equality attained? Does attaining 2q-4 in the general two-addition/one-deletion class force disjoint added edges between absolute points?
Falsifier: Any such incident example with <=2q-4 cycles, or any disjointness-criterion violation.
Competing explanation: A single deleted H-edge might destroy one single-added-edge C4 from each family plus one C4 using both added edges.
Dependencies: He–Ma–Yang arXiv:1912.00986v4, Lemmas 2.7 and 2.8; odd-order conic noncollinearity and uniqueness of polar line intersection.
Acceptance: complete ordinary proof plus two independent bounded integer checks of all incident absolute-edge pairs and all H-edge deletions for q=3,5. Domain sizes 288 and 5400; min expected 2q-3. Two sequential workers, compilation and execution each under hard 35-second timeout, no RNG. The finite data support but do not replace the all-q proof.
Stop after two worker attempts, or first falsifying certificate.
