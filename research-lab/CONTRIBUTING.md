# Contribute one checkable step

[Lab](README.md) · [Problems](problems/README.md) · [Progress](PROGRESS.md) · [Rules](AGENTS.md)

## The daily handoff

Read the actual remote head, the relevant problem page and its linked proof before choosing work. Keep one active question and one parked alternative. Choose a useful proof, counterexample, finite certificate or narrowly scoped formalization rather than a daily claim of discovery.

1. Fix the claim and falsifier. Use [the run template](templates/RUN.md).
2. Work in `research-lab/problems/<problem-id>/runs/YYYY-MM-DD-short-name/`. Keep proof, source and receipts together. Existing `studies/` paths remain historical and must not be moved casually.
3. Record exact scope, dependencies, actual commands, timeouts and negative results. Two bounded experiment workers remain the daily limit; this organization does not raise it.
4. Add the result to `problems/catalog.json`, update the relevant next question and update `STATUS.md` only when its claims or formal status change.
5. Run `python3 research-lab/tools/catalog.py --write`, then `python3 research-lab/tools/catalog.py --check`. Commit the generated indexes too.
6. Submit one coherent branch/PR. Inspect the exact commit's checks before claiming verification or promoting it. Never force-push the user's main branch.

## Three different kinds of verification

| Check | What it establishes | What it does not establish |
|---|---|---|
| Navigation/catalog check | Valid metadata, existing local links, indexed studies and deterministic pages | Mathematical correctness |
| Computational certificate | Only the declared finite domain and acceptance criteria | A universal theorem or literature novelty |
| Lean build + dependency audit | The intended Lean declarations at a pinned commit | Every claim described elsewhere in the repository |

Adding a program to the catalog does not execute it or automatically extend the mathematical CI suite. A new verifier needs a reviewed acceptance check in the verification workflow. Do not put shell commands in the catalog for CI to execute automatically.

`EXACT_FINITE_CERTIFICATE` is not `LEAN_VERIFIED_AT_COMMIT`. A Lean-verified entry must cite the precise Lean file, accepted commit and run evidence. Failed or suspended results stay labelled and visible.

## Visuals and licensing

Use short tables, descriptive link labels and expandable detail before adding decoration. CI and license badges have limited meanings. A diagram should link to its source and describe whether it is an illustration or certified data. Do not portray an unproved conjecture as a proved flowchart.

Original contributions in `research-lab/` use the existing [MIT license](LICENSE). Cited papers and third-party assets keep their own terms. No font files, correspondence, recipient lists, credentials or unpublished drafts may be committed.

## When a run is blocked

Preserve a local release candidate with the exact barrier. A scheduled run failing, a GitHub write failing and a proof failing are different events. Do not reuse an old PASS badge to conceal any of them. The GitHub workflow verifies submitted content; it does not independently invent or solve mathematics.
