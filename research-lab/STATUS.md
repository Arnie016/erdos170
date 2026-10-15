# Claim and formalization status

This is a public research notebook, not a list of solved headline Erdős problems. Literature priority is unchecked unless a study explicitly says otherwise.

| Claim | Scope | Evidence | Formal status |
|---|---|---|---|
| omega(G) ≤ 3 over all groups | h(1)=h(2)=1, h(3)=3; omega=3 iff G/Z(G) is Klein four | [Self-contained ordinary proof](problems/erdos-117/runs/2026-10-15-omega-three/PROOF.md); finite controls D8, Q8, S3, C4 | UNFORMALIZED |
| Order-64 examples | Two specified cocycles have a=omega=9 and a=omega=5 | Matching finite clique/cover witnesses; two executable checks | The explicit binary witness predicates are Lean-verified at e4c0bc8ac647538f5dd22c84613b6745446f7bfe; full group reduction not formalized |
| Binary central quotient dimension <=4 | All groups in that quotient class, including infinite centers | Ordinary reduction plus exhaustive 2,825-kernel certificate | UNFORMALIZED |
| Quotient dimension <=5, derived dimension <=2 | Exact stated group class | Ordinary reduction plus complete 174,251-plane certificate and lower-dimensional result | UNFORMALIZED |
| Modular n=6, N=69 | Complete signed unit-dyadic replacement-distance census | Two independent implementations, 66 canonical sets, 4,224 sign lifts | UNFORMALIZED |
| Basis-independent cross-radical bound | Exact hypotheses in October 5 note | Ordinary proof and finite matrix regression | UNFORMALIZED |
| General Erdős #117 | h(n)=sup a(G) over all groups with omega(G)<=n | No solution claimed by this project | OPEN IN THIS PROJECT |
| General Erdős #170 | Restricted difference bases and sparse-ruler asymptotics | Original suite outside this subtree retained; no solution claimed | OPEN IN THIS PROJECT |
| Earlier claimed recurrence improvements | Historical local/asymptotic calculations not imported here | Dependency audit needed before acceptance | NOT ACCEPTED AS GLOBAL THEOREMS |

## Verified versus reproduced

The initial Lean verification succeeded in [run 37905339322](https://github.com/Arnie016/erdos170/actions/runs/37905339322). Ten declarations have no axiom dependencies; the two cover checks use only Lean's standard propositional extensionality (`propext`). No admitted proof or native-evaluation axiom appears.

The study implementations were replayed from recovered source during publication. They have been ported to repository-relative output paths and edited for portability. Current-commit computational acceptance is provided by the dedicated CI job and its downloadable `SUMMARY.json`, not by this prose. The driver checks exact domains, histograms, agreement and relevant negative controls. A failed run means that version is not reproduced; retained historical counts must not override a failure.

## Active and parked questions

Active: is h(4)=4 over all groups? A single group with omega<=4 and a>=5 would refute the equality.

Parked: the rank-five, three-coordinate alternating-map clique-versus-cover gap. Do not promote a finite scan into a uniform group theorem.

## Vocabulary

Validity: ANALYTICAL_PROOF, EXACT_FINITE_CERTIFICATE, COUNTEREXAMPLE, CONDITIONAL_DERIVATION, CONJECTURE, SUSPENDED, RETRACTED.

Formal layer: UNFORMALIZED, LEAN_PENDING, LEAN_VERIFIED_AT_COMMIT.

Literature: KNOWN, PRIORITY_UNCHECKED, CANDIDATE_NOVELTY.

A complete finite result is not a solution beyond its quantified domain. Neither a local improvement nor a repository badge proves a new global bound.
