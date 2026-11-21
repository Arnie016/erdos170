# Verification that follows the claim

**Engineering update, 3 November 2026. No new theorem is claimed.** The new runner checks one registered certificate without replaying the historical archive. It does not send email, use the network, choose research tasks or modify acceptance criteria.

## Try it

```sh
python3 research-lab/harness/verify.py --list
python3 research-lab/harness/verify.py --only mod5-field47
python3 -m unittest discover -s research-lab/harness -p 'test_*.py' -v
```

The receipt is written to `research-lab/_generated/focused/SUMMARY.json`. It includes the selected claim, checker/input hashes, interpreter version, measured wall time, individual obligations and actual success/failure. The GitHub job records its commit through GITHUB_SHA. Local receipts do not invent a Git revision.

## Current coverage

| Check | Scope | Not covered |
|---|---|---|
| `mod5-field47` | Prime field, exact multiplicative order 47, five-term polynomial root, progression n=9+23k | Formal proof, novelty, all t=5 shapes |
| Harness tests | Corrupted constants, unknown verifier/IDs, unsafe paths, duplicate registrations and missing evidence | Universal security, private outreach, scheduler reliability |
| Existing `verify_all.py` | Its original eight historical studies | Automatically all newer dated studies |
| Existing Lean job | The actual declared Lean predicates | Arbitrary prose or this entire obstruction theorem |

See the [certificate proof](harness/PROOF-field47.md). Arithmetic is checked in two forms: modular exponentiation and separately accumulated repeated multiplication. These are differently structured checks inside one program, not independent researchers. The previously recovered research package remains the source for the proof; porting it is maintenance.

## How the harness may improve

A proposed change must name one bottleneck, preserve a frozen baseline and negative controls, then report exact behavior changed. Prefer a smaller inspectable certificate or reduced verification overhead to a larger search count. A new verifier type requires reviewed code and tests: registry JSON is data, never shell commands or imported plugins. Paths stay within the lab; invalid data fails closed. The runner cannot grant email, GitHub or compute permissions.

Future changes may add claim-dependency checks and explicit new-study CI coverage. Those capabilities are NOT implemented by this release. The current runner performs bounded arithmetic verification and records receipts. No claim of recursive model self-improvement is made.

## Publication

A public PR is not a merged result. Keep current-head checks, no-force-push rules and a stop on failed authorization/write safety checks. Main and open research branches remain distinct. The research job should not weaken its evaluator, broaden resource limits or silently auto-merge speculative claims. Vetted mathematical correspondence is handled separately with full history/opt-out checks; no recipient information belongs in this repository.

The [exploration board](EXPLORE.md) extends the owner's permitted selection beyond the old six-lane-only rule. The [design guide](DESIGN.md) describes how proof pictures remain honest and readable.
