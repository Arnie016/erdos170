# Open Mathematics Research Lab

[![Verification](https://github.com/Arnie016/erdos170/actions/workflows/math-research-ci.yml/badge.svg)](https://github.com/Arnie016/erdos170/actions/workflows/math-research-ci.yml)

Arnav Salkade's open, AI-assisted mathematics notebook: exact finite results, explicit counterexamples, reproducible programs and an incremental Lean formalization.

**This is not a claim to have solved Erdős #117 or #170.** Finite certificates, ordinary proofs and Lean-checked theorems are different achievements. Literature priority is separate from correctness.

This lab lives alongside the original sparse-ruler suite. Its experiment files and paper workflow remain unchanged. The badge displays the actual current workflow status, not a manually declared success.

## Explore the work

| Study | Precise scope | Proof and programs |
|---|---|---|
| Coupled-padding rank budget | Two restriction-rank inequalities; 16,896 matrix cases | [October 3](studies/erdos117-2026-10-03/proofs/coupled_padding_rank_budget.md) |
| Central-basis shear | A named scalar rank can change without changing commutation | [October 4](studies/erdos117-2026-10-04/proofs/central_basis_shear_counterexample.md) |
| Basis-invariant cross-radical envelope | A clique lower bound over the whole scalar-form space | [October 5](studies/erdos117-2026-10-05/proofs/basis_invariant_crossradical_envelope.md) |
| Two groups of order 64 | Explicit matching abelian covers and noncommuting sets of sizes 9 and 5 | [October 6](studies/erdos117-2026-10-06/proofs/order64_exact_cover_clique.md) |
| Ordered binary form pairs | All 4,096 maps on F2^4 with two scalar coordinates | [October 7](studies/erdos117-2026-10-07/proofs/order64_bilinear_family_census.md) |
| Every binary rank-four geometry | All 2,825 kernels in the exterior square; clique equals cover | [October 8](studies/erdos117-2026-10-08/proofs/all_rank4_pencils_census.md) |
| Binary rank-five, two-coordinate pencils | All 174,251 scalar planes; clique equals cover | [October 9](studies/erdos117-2026-10-09/proofs/rank5_two_coordinate_pencils.md) |
| Modular distinct subset sums | n=6, modulus 69: 4,224 admissible sets, each within one replacement of signed scaled powers of two | [Modular 69](studies/modular69/proofs/modular69_finite_theorem.md) |

All eight studies include source programs in their `src/` directories. Earlier #117 packages not recovered in this publication session are not silently represented as imported. The original [#170 project](../README.md) remains separate; its continuous autocorrelation parameter is not the discrete sparse-ruler limit.

## Reproduce

```sh
python3 research-lab/tools/verify_all.py
cd research-lab/lean
lake build
lake env lean OpenMath/Certificates.lean
```

The computational driver needs Python 3.10+ and g++ with C++17. It installs nothing, copies source into a fresh directory and runs fixed finite domains. Each process has a 35-second timeout; raw outputs, failures and source hashes remain in `_generated/`. Slower machines may time out; partial output is not promoted to a full certificate.

The Lean project pins 4.19.0 and uses Std only. The initial 12 declarations passed the [recorded Lean CI run](https://github.com/Arnie016/erdos170/actions/runs/37905339322). **The large censuses and the abstract group-to-graph reduction are not formalized in Lean.**

## Contribute

Read [STATUS.md](STATUS.md), [AGENTS.md](AGENTS.md) and [REPRODUCING.md](REPRODUCING.md). A useful contribution is one exact claim with a proof or independently checked certificate and an honest boundary. Conjectures and rejected approaches belong here when clearly labelled.

The next finite question is whether a three-dimensional scalar pencil on F2^5 has a chromatic gap. No all-dimension assertion follows from the current censuses.

## Publication policy

Only original mathematical notes and safe verification programs are imported. Private correspondence, contact ledgers, unpublished third-party manuscripts and credentials are excluded. Referenced papers retain their attribution and terms. Source ports replace absolute sandbox outputs with repository-relative paths; prose has been edited for public scope and evidence clarity. The CI run on each exact commit determines reproduction status.

Human project owner: Arnav Salkade. AI assists with exploration, implementation and drafting. Automated verification is not independent human peer review or institutional endorsement. Original contributions in this subtree are available under the [MIT license](LICENSE).
