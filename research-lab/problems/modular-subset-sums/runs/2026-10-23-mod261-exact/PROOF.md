# Complete eight-element classification modulo 261

Validity: EXACT_FINITE_CERTIFICATE with an ordinary proof of complete reduction. Literature priority: UNCHECKED for finite completeness; perturbation constructions are KNOWN. Lean: UNFORMALIZED.

## Statement

There are exactly 64,512 eight-element subsets A of Z/261Z with all 256 labelled subset sums distinct. Up to multiplication by a unit and independent signs, exactly three classes occur:

- {1,2,4,8,16,32,64,128};
- {1,2,4,8,16,32,64,129};
- {1,2,4,8,16,32,65,129}.

Every admissible set overlaps a signed unit dilation of the powers of two in at least seven members. Exactly 21,504 are dyadic; the remaining 43,008 have maximum overlap exactly seven. This is a finite N=261 result, not a general Erdős distinct-sums solution.

## Why the normalization is complete

An admissible set contains neither zero nor opposite members. Sign changes translate its subset-sum set; unit multiplication is invertible. Both preserve injectivity.

For a in A, put g=gcd(a,261). Pair subsets without a with those containing a. Within each residue class modulo g the number of represented residues is even, while each full fiber has odd size 261/g. Thus at least one residue is missing per fiber. Only five residues are missing, so g<=5. As 261=9*29, gcd(a,261) is 1 or3.

If two members were divisible by3, the occupied count in each residue class modulo3 would be divisible by4. Each fiber has87 residues and can then contain at most84 subset sums, giving at most252 overall, not256. Hence at most one member is a nonunit. Scale one of the at least seven units to1 and sign-normalize every member into1,...,130.

The complete reduced domain is therefore increasing eight-subsets containing1, excluding multiples of9 or29 and allowing at most one multiple of3. No other arithmetic or heuristic pruning is used. This explicitly includes the nonunit cases.

## Exact certificate

Worker A maintains subset sums S and accepts a new mark a iff S and S+a are disjoint. Worker B maintains D=S-S, accepts a iff a is absent from D, and updates to D union(D+a)union(D-a). Both traverse the complete reduced domain. The second method separately reconstructs all256 subset masks at each accepted leaf using an ordered set.

The two complete accepted lists agree. Depth counts0,...,8 are [0,1,106,5230,151955,2675386,17477423,5418406,22]; depth0 is zero because both begin with the normalized1. Expanding these22 leaves by all168 units and sign-normalizing yields252 canonical sets. Each is directly verified and all are partitioned into three unit orbits of84. Exact maximum dyadic overlap histogram: seven for168 sets, eight for84. There are84 all-unit canonical sets and168 with one nonunit.

Each canonical set has256 distinct sign lifts and different canonical sets have disjoint lifts. Thus252*256=64,512. A missing-leaf list is rejected; the colliding control {1,2,3,4,5,6,7,8} has37 subset sums. Both rotation primitives are independently tested on all single-bit inputs and nonzero shifts.

The first representative is dyadic. The second replaces128 with129. Multiplying powers of two by65 modulo261 and choosing signs gives {1,2,4,8,16,32,65,130}; replacing130 with129 gives the third representative. These witnesses transfer under signs and units. The enumeration proves that the latter two classes have no full dyadic overlap.

## Provenance and limits

The prior n7 classification is at b1f64b9577e67856c277c26d27c0728556a4d4a3; it was read, not rerun. The templates and sign/unit symmetries are published background: Cambie, Gao, Kim and Liu, The Erdős distinct subset sums problem in a modular setting, arXiv:2308.03748 (2023), Acta Arithmetica217(2025),295-307, DOI10.4064/aa231107-13-9, Lemmas2.1-2.2, Section7 and PropositionA.1. The gcd argument extends their residue-parity method. Finite-completeness priority is unchecked. No Lean proof or universal classification is claimed.

Next useful question: can the five centered missing residues be forced into the three template shapes for general n? An admissible set outside those shapes is a falsifier; absence in this finite census is not a universal argument.
