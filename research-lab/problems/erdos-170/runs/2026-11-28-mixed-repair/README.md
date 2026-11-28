# Mixed deletion baseline

The match began after the one-middle, one-exterior deletion result had been proved. It earns zero points in the duel. Its proof derives the collision table used by the later arbitrary-pair result. [Frozen statement](STATEMENT.md) · [Proof](PROOF.md)

From this directory, the following commands check the designated unique edges and replay the separate finite search through radius 24.

    python3 -B verify_edges.py
    python3 -B verify_mixed.py --max-r 24

The edge checker covers radii 1 through 32 and radius 65. The finite mixed search covers 39,152 deletion pairs and includes the two earlier strict-middle deletion controls. The generic one-new-mark inequality and the statement comparison build in the pinned Lean project. The mixed obstruction itself has an ordinary proof. [Pinned project](../../../../lean) · [Published Wichmann completeness theorem](https://www.combinatorics.org/ojs/index.php/eljc/article/download/v33i1p40/pdf/)
