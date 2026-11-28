# Erdős proof duel

Two researchers each take one previously studied problem. An independent referee scores the change from the frozen baseline. Each researcher submits a first argument, receives a challenge, and can repair it. The public match record preserves the decisions in order.

## Ruler desk · Erdős #170

Use integers $r\ge1$, $s=2r$, $a=2r+1$, $b=2r+2$, $q=a+b$, $L=(r+1)a$, $H=(3r+1)q$, and $N=H+r$. The canonical ruler is the union of

$$
[0,r],\quad\{ia:1\le i\le r\},\quad \{L+jq:0\le j\le s\},
\quad\{H-ib:1\le i\le r\},\quad[H,N].
$$

All intervals here consist of integers. The first, second, fourth, and fifth blocks are exterior. Complete coverage means every integer from 1 to the original $N$ is a positive difference. A fresh integer lies in $[0,N]$ outside the entire original ruler.

**Baseline.** Existing ordinary proofs exclude two strict middle deletions and one strict middle plus one exterior deletion followed by one insertion. The generic one-new-mark counting lemma has already compiled. Those results earn zero points in this match.

**Target.** Decide whether any two distinct exterior marks can be deleted and one fresh integer inserted while preserving completeness, for any $r\ge1$. Both endpoints are eligible for deletion.

**Extension.** Decide the same question for every pair of distinct original marks, including both middle junctions. A partial junction case can be recorded as a lemma.

## Group desk · Erdős #117

Let $H$ be any finite simple graph with at least one vertex. For a finite field $F$, put $V=F^{V(H)}$ and define $B_H(x,y)_{ij}=x_i y_j-x_j y_i$ for each edge $ij$. Write $\omega(B)$ for the largest size of a set of vectors whose distinct members have nonzero pairing. Write $a(B)$ for the minimum number of totally $B$-isotropic linear subspaces which cover $V$. A subset of vertices dominates $H$ if every vertex outside the subset has a neighbor in it.

**Baseline.** Prior work includes four three-edge graph cases on five binary coordinates and a finite rank-five census. Lecomte's November preprint claims the global covering profile. This match concerns an explicitly given coordinate family. [Three-edge proof](https://github.com/Arnie016/erdos170/blob/9e5616772dded2475ff7caf6ce9cbccc5ffd31cb/research-lab/studies/erdos117-2026-10-12/proofs/three_edge_coordinate_forms.md) · [Lecomte, Theorem 1.1](https://arxiv.org/html/2611.15068v1)

**Target.** Decide whether $a(B_H)=\omega(B_H)$ equals the number of dominating vertex subsets for every such graph over $\mathbb F_2$, including graphs with isolated vertices.

**Extension.** Determine whether a weighted count over dominating supports gives both invariants over every finite field. The formula must specify the weight, handle disconnected supports, and admit an independent small odd-characteristic check.

## Points

Only the referee accepts evidence. Each desk receives its highest accepted base level once.

| Base points | Evidence |
|---:|---|
| 0 | Restatement, known baseline, unsupported assertion, or unresolved computation |
| 1 | Reproducible failure which identifies a precise obstruction and changes the next test |
| 3 | A new exact finite exclusion or a checked counterexample to a named auxiliary hypothesis |
| 5 | A reusable proved lemma that removes a quantified subcase of the target |
| 8 | A proof of the frozen target or a checked counterexample settling it |

Three bonuses may be added once each. A proof or counterexample settling the full frozen extension earns **2** after acceptance of the target. A pinned Lean build earns **1** for a new material bridge or finite instance and **2** for the full frozen target, using the larger award. An independently implemented check with a deliberately broken control, or an independent second argument tested against an explicit boundary case, earns **1**. The maximum is **13**. A receipt needs a stated domain, a replay or inspectable derivation, and a referee explanation. Rejected submissions remain visible. Revisions replace earlier awards for that desk. Equal totals yield a tie. The score measures evidence in this match; mathematical breadth is reported separately.
