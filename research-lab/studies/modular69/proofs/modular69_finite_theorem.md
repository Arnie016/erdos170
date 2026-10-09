# Modular distinct subset sums: n=6, N=69

Status: exact finite certificate, with an ordinary proof of symmetry reduction. Literature priority unchecked. No all-n theorem is asserted.

## Statement

Let A be a six-element subset of nonzero residues modulo 69, with all 64 subset sums distinct (including the empty subset). Then A shares at least five elements with some signed unit dilation of {1,2,4,8,16,32}.

There are exactly 4,224 admissible A: 1,408 are themselves signed unit-dyadic sets, and 2,816 have maximum overlap exactly five.

## Complete reduction

A cannot contain zero or both x and -x. Replacing one entry a by -a translates the subset-sum set: if S comprises the sums of other entries, `S union (S-a) = (S union (S+a))-a`. It preserves injectivity, by a bijection of the subset masks.

As 69 is odd, every sign pair has a unique representative in 1,...,34. Each admissible set therefore corresponds to a unique six-element canonical set in this interval and has exactly 64 distinct sign lifts. Different canonical sets have disjoint lifts.

The maximum overlap with the signed dyadic family is invariant under these sign changes. For a unit u, the six pairs ±u2^i are distinct, and any matching sign pair can have its sign chosen independently.

It therefore suffices to cover all C(34,6)=1,344,904 canonical candidates instead of C(68,6)=109,453,344 unnormalized candidates. A prefix collision persists on extension, so collision pruning excludes no valid completion.

## Independent algorithms

The generator uses exact 69-bit cyclic rotations to extend increasing prefixes. It writes every accepted canonical set with a unit-dilation overlap witness.

The second program reconstructs the entire census using ordinary sets of residues, compares complete lists, explicitly generates all signed unit-dyadic sets and directly sums all 64 subset masks for every accepted sign lift. It verifies the exact maximum overlap and rejects a missing-candidate list and the colliding set {1,2,3,4,5,6}.

There are 66 canonical sets: 22 have overlap six and 44 have overlap five. Multiplying by 64 gives the counts in the statement. The signed dyadic orbit contains 1,408 sets.

A simple witness requiring one replacement is {1,2,4,8,16,33}. Its subset sums are {0,...,31} union {33,...,64}; replacing 33 by 32 yields the standard dyadic set. The census confirms maximum overlap five, not merely at least five.

## Reproduce

```sh
mkdir -p results
python3 src/generate.py results/candidates.json
python3 src/verify.py results/candidates.json results/verification.json
```

The source implementations use the Python standard library. The universal mathematical claim for N=2^n+5 remains unresolved by this computation; n=7 is outside this certificate. This study is distinct from Erdős #117 and #170.
