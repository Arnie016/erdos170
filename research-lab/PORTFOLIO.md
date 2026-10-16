# Mathematics portfolio: one worthwhile question per run

**Selection policy updated 16 October 2026 at the project owner's request.**

This is a research plan, not a mathematical result. The daily task may select any of six lanes. There is no permanent priority for Erdős #117. Existing ACTIVE, RETAINED and PARKED catalog labels record earlier focus; they do not prohibit selecting another lane under this policy. Existing evidence and mathematical claim status are unchanged.

[Existing results](problems/README.md) · [Claim status](STATUS.md) · [Reproduction](REPRODUCING.md) · [Contribution rules](AGENTS.md)

## The six choices

| Problem | Plain-language question | A worthwhile next research unit |
|---|---|---|
| **Erdős #117: group covers** | How many commuting pieces cover a group? | Decide a remaining structural or finite-threshold question after reading current source corrections, rather than repeating completed small cases. |
| **Erdős #170: sparse rulers** | How few marks can measure every distance? | Test coordinated repairs or a precise scalable construction obstruction. A complete candidate improves a saved baseline, not automatically the literature record. |
| **Modular distinct subset sums** | How different can an admissible set be from signed powers of two? | Seek a certified modulus-133 counterexample to the one-replacement pattern, or prove a structural lemma. Keep the modular branch separate from integer distinct-subset-sum conjectures. |
| **Erdős #242: three unit fractions** | Can every 4/n, n>2, be written as 1/x+1/y+1/z with positive x<y<z? | After checking known identities, attempt a symbolic formula for an entire declared arithmetic progression, or prove that a sharply bounded denominator-template family fails. |
| **Erdős #30: Sidon sets** | How densely can numbers fit without repeated pair sums? | Find a quantitative boundary/correlation improvement in a diameter argument for a precisely stated class, or an exact counterexample to the proposed inequality. |
| **Erdős #60: unavoidable four-cycles** | Once a graph exceeds the maximum edge count without a square, how many squares must occur? | Test a precise edge-switch/cycle-count lemma in an uncovered regime, accounting for published even-order results and the actual extremal edge threshold. |

The final three lanes are newly added research choices. No result in them is claimed by adding this page. A new problem page and catalog entry should record the real selected question and source baseline when that lane is opened, not pretend that an unstarted project is completed.

## Selection and stopping rules

Each round chooses **one** question and makes one attempt with a predeclared falsifier and acceptance test. Consider at most two serious candidates. Select by remaining mathematical importance, ability to distinguish competing explanations and feasibility of verification, not by the number of cases an easy program can print.

Avoid more than two consecutive rounds on an unchanged tactic without a new implication, witness or unresolved proof step. After two no-advance attempts on the same tactic, park it and select another lane. Across roughly six substantive rounds, aim to visit at least three lanes when useful accessible questions exist. This is not a forced rotation or a discovery quota; a near-finished genuine proof can justify a longer streak.

Every result should answer **what uncertainty was decided and why that changes the parent problem**. Rebuilding a known certificate, refreshing paper timestamps or adding folders is maintenance, not a breakthrough. Negative results count when they rule out a stated mechanism. Finite searches retain finite scope. An unchanged global bound must remain explicitly unchanged.

The two-worker limit, hard timeouts, exact checking, source provenance, privacy and guarded GitHub publication rules remain in force. No six-way parallel search, increased compute allowance, new scheduled task or automatic outreach campaign is authorized by this change. The weekly Observatory is separate.

## Source baselines and reading depth

These are starting sources, not claims that their entire proofs have been independently audited. Check current versions before using a theorem or claiming an improvement.

### Erdős #242

[Canonical statement](https://www.erdosproblems.com/242): for every n>2, seek positive integers x<y<z satisfying 4/n=1/x+1/y+1/z.

Spiridon Mihnea and Dumitru C. Bogdan, *Further verification and empirical evidence for the Erdős-Straus conjecture*, [arXiv:2509.00128v1](https://arxiv.org/abs/2509.00128v1), submitted 29 August 2025. Reading depth for this update: primary abstract and version metadata. The authors report verification through 10^18. Therefore a tiny numerical scan is not a new frontier; symbolic families, precise obstructions and useful proof certificates are better initial targets. Integrality, positivity, distinctness and residue-class coverage are proof obligations.

### Erdős #30

[Canonical statement](https://www.erdosproblems.com/30): if F(N) is the largest size of a Sidon subset of {1,...,N}, is F(N)=sqrt(N)+O_epsilon(N^epsilon) for every epsilon>0? A Sidon set has no nontrivial a+b=c+d, with unordered pairs and repetitions defined consistently.

Daniel Carter, Zach Hunter and Kevin O'Bryant, *On the Diameter of Finite Sidon Sets*, [arXiv:2310.20032](https://arxiv.org/abs/2310.20032), [Acta Mathematica Hungarica 175, 108–126 (2025)](https://doi.org/10.1007/s10474-024-01499-8). Reading depth: publisher/arXiv abstract and bibliographic metadata. Their stated upper bound is sqrt(N)+0.98183 N^(1/4)+O(1), after matching interval conventions. Do not rediscover an older coefficient or conflate Sidon nonrepetition with the sparse-ruler requirement of covering all distances.

### Erdős #60

[Canonical statement](https://www.erdosproblems.com/60): a simple n-vertex graph with more than ex(n,C4) edges is conjectured to contain at least c sqrt(n) four-cycles in the asymptotic regime, for an absolute c>0. Here ex(n,C4) is the largest edge count of a C4-free n-vertex graph. Cycles are not required to be induced.

Jialin He, Jie Ma and Tianchi Yang, *Some exact results on 4-cycles: stability and supersaturation*, [arXiv:1912.00986](https://arxiv.org/abs/1912.00986), [journal DOI](https://doi.org/10.4208/csiam-am.SO-2021-0006). Reading depth: abstract and introductory theorem statements, including Theorems 1.6–1.9. Significant large-even-q cases are already settled. An odd-order or small-parameter experiment must still check whether other literature covers it. A guessed value of ex(n,C4) cannot support a claimed counterexample. Problem #60 here is not the power-of-two-cycle problem.

### Status caveat

The canonical problem records were accessible through search indexing during this update, but direct fetches failed. This page does not assert a freshly verified October-2026 global open/solved status. Source checks are required before substantive work in a new lane. Search discovery and abstract reading are not full proof verification.

## What remains unchanged

[Existing #117 work](problems/erdos-117/README.md), [#170 work](problems/erdos-170/README.md) and the [modular branch](problems/modular-subset-sums/README.md) retain their original scope. No historical theorem is promoted by this planning update. Lean verification remains limited to the statements actually checked. Ordinary proofs, computational certificates and source-reported results stay separate.

Version new substantive work under its problem, retain failures and report actual commit/CI evidence. No new mathematical experiment was run to add this portfolio. Original lab material remains under the [scoped MIT license](LICENSE).
