# Execution receipts | 11 November 2026

Frozen complete domain: \(r=3,q=6,t=18,N=35\). Each \(a\in\{1,\ldots,8\}\), each right reflected six-subset of positions 0 through 17 containing 0. Total \(8\binom{17}{5}=49,504\). No random samples or seeds.

Frozen acceptance: exactly four tail-covering configurations, at \(a=1,2,3,6\), with the right blocks specified by the theorem; both checkers reject a corrupted a=3 profile.

| Program | Integer method | Locally measured wall | Exit | Certified profiles |
|---|---|---:|---:|---|
| worker_bits.py | Sumset bit shift / union | 1.94 s | 0 | 4 of 49,504 |
| worker_differences.py | Direct ruler coordinates and unordered difference multiplicities | 1.44 s | 0 | same four |

Each executed sequentially as a separate `timeout -s KILL 35s python3 <script>` process in Linux Python 3.13.5. No failure, restart or background job. Source SHA256 hashes from the executed sandbox files: worker_bits.py 7f38677ba06f1259881492c1c3c787046af16156e9ef8cde30233ceeb8526d6c; worker_differences.py 3b81ad96ab4419fe5d239f46d30ebd1cd86d48f941f441ee2cc3e91118f6ef4d.

Expected profiles (a : R):
- 1 : {0,3,6,9,12,15}
- 2 : {0,1,6,7,12,13}
- 3 : {0,1,2,9,10,11}
- 6 : {0,1,2,3,4,5}

Local raw stdout/stderr and hashes are retained in the downloadable accompanying 11 November evidence ZIP. The scripts themselves check complete domains and exact output, not a solver status alone.

**Remote reproduction:** dedicated GitHub Actions CI must be observed at the resulting commit before claiming CI passed. A historical green job is not evidence that the newly added checkers ran. The analytical theorem is not Lean verified.
