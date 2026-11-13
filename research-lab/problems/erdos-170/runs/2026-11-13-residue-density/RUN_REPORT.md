# Verification report | 13 November 2026

The 12 November ordinary proof stood independently of computation, but its second checker failed a **false negative-control expectation**: M(3,12,4)=4, not >4. The correction tests M(3,12,5)=5>4 instead. The independent checker used today was extracted from the retained 12 November corrected candidate, with no algorithmic change. The published version additionally fixes an obsolete explanatory comment only.

Two processes, sequential, each under `timeout -s KILL 35s`, no retries:

| Worker | Exit | Measured wall | Exact source SHA-256 locally executed |
|---|---:|---:|---|
| `worker_formula.py` | 0 | 574 ms | `836b28f39873333b72a89f86a5b7d89f2b58121afe8231b2855b6ad43d8f43dc` |
| `worker_independent_corrected.py` | 0 | 708 ms | `ad777eb818075d6269a4ad66931520ea87d6d30287952b4a1bf05041453706a8` |

Both produced the same fourteen exact minimum counts. The independent worker enumerated every right-tail subset (4096 masks) for each of five small profiles and gave [4,4,6,4,5] with positive witness counts. Negative controls passed. The original 12 November failure is retained in the local source archive, not erased.

**Caveat:** `worker_independent.py` committed here contains a one-line **comment-only** clarification compared with the executed corrected file; computational logic is identical. Dedicated GitHub CI is requested to execute the published exact bytes. Until its status is observed, the CI outcome must be described as PENDING, not PASS.

**Reproduction:** `timeout -s KILL 35s python3 worker_formula.py > worker_formula.ci.json`; `timeout -s KILL 35s python3 worker_independent.py > worker_independent.ci.json`; `python3 verify_receipts.py`.

No outreach, full ruler optimality statement, new global result or formalized proof is claimed.
