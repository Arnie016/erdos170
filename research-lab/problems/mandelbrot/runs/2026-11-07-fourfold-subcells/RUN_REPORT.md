# Local execution | 7 November 2026

Predeclared complete domain: nine previously uncertified parents, each split into 4×4 children (144 squares). Predeclared acceptance: strict rational enclosure proving |z₆(c)|>2 at every point of a closed child. Two sequential workers, hard limit 35 seconds each, no retries or random seed.

| Worker | Executed source SHA256 | Exit | Measured wall | Individual accepted |
|---|---|---:|---:|---:|
| worker1.py, centered Gaussian-rational interval Horner | 4e5e4c434f972b008c9ab4c6827665a6d5bdf8aaeae816634c978fbbc2c1ed35 | 0 | 1.26 s | 105 |
| worker2.py, independently generated integer polynomial and binomial translation | 1b88b87e4cc6cf1b5f8ff38c2489a276b3a0f131604180b7c5a55ae371f31852 | 0 | 1.28 s | 102 |

Commands: /usr/bin/time -f 'elapsed_seconds=%e cpu_seconds=%U exit=%x' timeout -k 1s 35s python3 worker1.py >worker1.json 2>worker1.stderr, then same for worker2.py. Worker 2 computes its own list before reading the first JSON to intersect. 102 jointly accepted, three worker1-only: (3,-9,3,1),(6,-7,2,3),(6,-6,3,0). Local raw JSON and shell receipts are retained in the downloadable dated research package.

Negative controls: c=0 not accepted as escaped; c=c₀ accepted by worker1. No paid resources or GPU. Dedicated new-study GitHub CI has **not** been observed, and no Lean proof is claimed. The source SHA256 values refer to locally executed files; compare Git blob hashes before asserting the published files are identical.
