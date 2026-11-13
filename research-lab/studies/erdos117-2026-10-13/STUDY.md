# Erdős #117: a complete fixed-plane extension slice in binary rank five

**Date:** 2026-10-13 (Asia/Singapore)  
**Validity:** EXACT_FINITE_CERTIFICATE (local execution) + ANALYTICAL_PROOF (restricted group reduction)  
**Literature priority:** PRIORITY_UNCHECKED  
**Formalization:** UNFORMALIZED. This specific result has not run through repository CI or Lean.

## Claim contract

Let V=F₂⁵. Order the basis wedges of Alt(V) as 01,02,03,04,12,13,14,23,24,34. Fix the scalar alternating forms

A=e₀*∧e₁*+e₂*∧e₃* (10-bit mask 129),
B=e₀*∧e₂*+e₁*∧e₄* (mask 66).

For **every** C outside P=span_F₂{A,B}, define F_C(x,y)=(A(x,y), B(x,y), C(x,y)). Let Γ_C have 31 vertices V\{0}, adjacent iff F_C(x,y)≠0. The checked proposition is χ(Γ_C)=ω(Γ_C).

The radical of A is span(e₄), and that of B is span(e₃). Consequently their common radical is zero: this is a faithful base two-plane, not a restriction-induced radical.

A different C defines the same 3-dimensional scalar span if and only if it lies in C+P. There are 2^(10−2)−1=**255** nonzero cosets of P. Canonical C is the least integer in {C,C xor 129,C xor 66,C xor 195}. This is a complete 255-member finite domain, **not** 255 pairwise nonisomorphic groups.

**Falsifier:** a canonical C for which an exact lower bound χ(Γ_C)>ω(Γ_C) is demonstrated.  
**Competing explanation:** ω(A,B)=9, so this slice may bypass smaller-clique and genuinely harder pencils.

## Exact local certificates and independent check

Worker 1 generated every Γ_C by exterior-product masks, computed maximum clique by bitset branch-and-bound, and checked optimal colorability by exact DSATUR. Worker 2 independently reconstructed the bilinear graphs from matrix coefficients, computed maximum clique using Bron–Kerbosch, independently enumerated the **374** vector subspaces of F₂⁵, and constructed a matching cover using actual isotropic subspaces. It verified all coset keys, clique masks, subspace isotropy and point coverage.

A clique with k elements forces any abelian cover to use ≥k subgroups. A family of k totally isotropic subspaces whose nonzero points cover V\{0} supplies ≤k abelian subgroups. For each of the 255 instances these certificates matched.

| k=ω=χ | Number of pencils |
|---:|---:|
| 9 | 4 |
| 11 | 6 |
| 13 | 69 |
| 14 | 112 |
| 15 | 40 |
| 17 | 24 |
| **Total** | **255** |

### Compact per-pencil values

The next string has **255** characters, one for each canonical C in increasing numerical order. Map characters `9→9, B→11, D→13, E→14, F→15, H→17`. For every C, both ω and χ equal the decoded integer.

```text
FFDHDHDHHDDDDDDDFFDHFHFHHFFEEEEBFFDHDHFEEEEEEDDBFFDHFHDEEEEDDEEBFFDEEEEHHDFEDEDBFFDEEEEHHFDDEDED99BEEEEEEEEDEEDD99BEEEEEEEEEDDEFFFEDDEDDEDDHEEDEEEFEEDEEDEEHEEFFFFEDDEDEDDDEHDEEEEFEEDEDEEEEHFEFFFEEDDDDEDDEDHEEEEFDEEEEDEEEFHEEEEFDEEEDEEEFEEHFFFEEDDDEDDDDEEH
```

Local raw receipts: worker1 CSV SHA256 `c93188a490aea82fa7c31068d03ad751d9439c930621ec09ecbb05dbc27786d3`; worker2 exact clique/isotropic-cover CSV SHA256 `552c3b02aeb366519c7c35081659210c40e4669ed4df05e7918797456ea5a985`. Complete source, raw witnesses and logs are in the dated task-chat ZIP; **they have not been copied into GitHub**. The GitHub note is not a standalone independently reproducible certificate until that source/data port is validated. Do not treat existing CI green checks as reproducing this result.

## Restricted group consequence

Suppose a possibly infinite group G satisfies G/Z(G)≅F₂⁵, dim_F₂ G'=3, and the three-dimensional space of scalar forms induced by the commutator contains a GL(V)-conjugate of P. Changing the vector basis and choosing the corresponding three output coordinates identifies its commutation relation with some F_C. Every commuting set spans a commutator-isotropic subspace of V, whose full inverse image in G is abelian. Conversely every abelian subgroup projects into an isotropic subspace. Thus a(G)=χ(Γ_C) and ω(G)=ω(Γ_C), including infinite centers. **Therefore a(G)=ω(G) for this restricted class.**

This result does **not** exclude a chromatic gap for all rank-five, three-coordinate maps, improve a uniform constant, determine the eventual threshold n₀ or change the all-group h(n).

## Verification receipts and limitations

Both experiment workers were run under separate 35-second hard timeouts and exited 0. Worker 1 completed 255 cases; its actual wall time was not independently measured. Worker 2 completed 255 cases in 0.02 seconds (recorded wall time). No third worker, GPU or paid compute. A future repository integration should import the exact checked implementations and raw witnesses, freeze checksum checks and rerun both algorithms with negative controls before upgrading the *repository-CI* status.

Historical context: https://www.erdosproblems.com/117 ; Guillaume Lecomte, *Sharp Asymptotics for Abelian Covers of Groups with Bounded Noncommutativity*, arXiv:2608.20507v1. Today's result depends on neither a private manuscript nor the old global recurrence.

**Next discriminatory test:** a complete 255-extension slice for a second GL(5,2) orbit of faithful two-planes with smaller baseline clique number, or a checked explicit gap. No claim of generality beyond this P has been proved.
