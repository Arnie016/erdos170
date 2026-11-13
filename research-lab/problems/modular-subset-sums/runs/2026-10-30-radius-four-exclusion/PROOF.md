# Exact finite exclusion of a five-hole shape at N=517

**Claim MOD5-R4-517-20261030:** EXACT_FINITE_CERTIFICATE; literature PRIORITY_UNCHECKED; formalization UNFORMALIZED. This is not an all-n theorem.

**Statement.** No nine-element set A modulo 517 can have all 512 labelled subset sums distinct and exactly five missing residues of the form {c,c±1,c±4}, for any residue c.

**Proof, conditional on two independent complete integer enumerations.** Translate the represented sum set by -c. This normalizes the holes to H={0,1,516,4,513} and the represented set to S=(Z/517Z)\H. We do not assume that the translated S contains zero. Each actual generator a has a unique half-set B_a with S=B_a disjoint-union (B_a+a) and |B_a|=256.

All genuine generators are units: a nonunit has gcd(a,517)=11,47 or517, so addition by a splits the odd-order cyclic group into at least 11 odd cycles. Every such cycle needs a missing residue, otherwise binary alternation on a hole-free odd cycle is impossible. Five holes cannot meet 11 cycles. Thus exactly the 460 units form the complete candidate domain.

A unit a passes the necessary one-generator test exactly if all successive cyclic distances between holes, measured in steps of +a, are odd. Then B_a is uniquely recovered by alternation. Equivalently, the binary indicators satisfy f(x)=b(x)+b(x-a), which the independent implementation solves directly.

If a,b are two different genuine generators, C=S(A\{a,b})-c has 128 residues and S is the disjoint union of C, C+a, C+b and C+a+b. Uniqueness forces C=B_a∩B_b. Build a graph with qualifying units as vertices and edges exactly when this 128-point intersection generates four disjoint translates covering S. Every actual nine-element A would yield a 9-clique in this **necessary** graph; even a 9-clique would not suffice to produce A.

Two independent exact programs exhaust all 460 unit candidates. Their full sorted vertex and edge lists agree: 132 vertices and 40 edges. Of the 132 vertices, 96 have degree zero. The remaining 36 have this explicit bipartition:

L={3,21,36,39,58,89,92,119,127,390,398,425,428,459,478,481,496,514}

R={12,18,42,116,178,184,195,238,239,278,279,322,333,339,401,475,499,505}

All 40 exact edges (recorded in results/graph.json) cross from L to R. None connect two vertices in the same part. Hence the graph is bipartite and triangle-free, with maximum clique size 2 (edge 3,12 shows equality). A genuine nine-generator set requires a 9-clique, contradiction. QED.

**Independent control:** The known realizable centered shape {0,±1,±3} yields exactly 156 qualifying vertices, 308 compatible edges and clique number 9 under both methods, matching the 27 October control. For the new shape {0,±1,±4}, the two algorithms (geometric hole alternation with C++ bitsets versus direct binary recurrence with Python sets) agree on every listed vertex and edge. Both finish under separate 35-second hard timeouts. The proof is finite and computationally certified, not Lean-formalized.

**Limitation:** This excludes one centered shape at n=9, and its shifts and invertible dilations; it does not prove a classification of every five-hole shape or any unbounded n-family. Published signed/unit equivalences and known perturbations are in Cambie–Gao–Kim–Liu, arXiv:2308.03748, Acta Arithmetica 217 (2025). The 26 October inverse-hole theorem is an existing prerequisite, not a discovery claimed here.
