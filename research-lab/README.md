# Open Mathematics Research Lab

[![Verification](https://github.com/Arnie016/erdos170/actions/workflows/math-research-ci.yml/badge.svg)](https://github.com/Arnie016/erdos170/actions/workflows/math-research-ci.yml)
[![License: MIT in research-lab](https://img.shields.io/badge/license-MIT%20in%20research--lab-blue)](LICENSE)

**Explore a problem. Inspect the evidence. Continue from a precise question.**

| Start here | What you will find |
|---|---|
| [Erdős #117](problems/erdos-117/README.md) | Active group-cover research; finite results and the next unresolved case |
| [Erdős #170](problems/erdos-170/README.md) | Original sparse-ruler suite, retained as a separate lane |
| [Erdős #242](problems/erdos-242/README.md) | Exact minimum first denominator for an infinite composite family; not the global conjecture |
| [Modular subset sums](problems/modular-subset-sums/README.md) | Certified modulus-69 case and the parked modulus-133 question |
| [Progress log](PROGRESS.md) | Dated mathematical entries, with infrastructure updates separated |
| [Verification guide](REPRODUCING.md) | Exact checkers, CI receipts and Lean scope |
| [Contribute](CONTRIBUTING.md) | One bounded claim, reproducible evidence and a reviewable change |

> A green build is not a headline solution. The large censuses are not yet Lean proofs.
> See [claim status](STATUS.md) for validity and formalization, not just a badge.

## Evidence, not just results

The [problem directory](problems/README.md) is generated from [catalog.json](problems/catalog.json).
Each result points to its proof and executable source. Historical study paths are retained, so existing links and checkers keep working.

The [Lean library](lean/README.md) currently checks small explicit witness predicates. Its theorem scope is narrower than the ordinary group arguments and larger computational censuses.

## Keep the notebook current

```sh
python3 research-lab/tools/catalog.py --write
python3 research-lab/tools/catalog.py --check
```

Register a meaningful contribution, regenerate the indexes, then submit a reviewed commit or pull request. CI checks navigation separately from mathematics. Scheduled chat runs are research attempts, not guaranteed discoveries or guaranteed repository access.

Only original research and safe verification artifacts belong here. Private correspondence and unpublished third-party material stay out. The [MIT license](LICENSE) applies to original material in this subtree, not automatically to the older root project or cited papers.
