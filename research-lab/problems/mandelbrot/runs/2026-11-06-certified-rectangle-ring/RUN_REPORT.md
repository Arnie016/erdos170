# Execution receipt | 6 November 2026

**Frozen domain:** integer grid i,j∈{−10,…,9}, 400 possible 0.001-sized squares; exact whole-annulus filter produces 216 eligible. **Success:** both exact rational methods certify the same 207; 9 are not certified. No sampling, random seed, external library or GPU used. **Limits:** sequential processes, each under hard 35-second timeout, no retries.

| Process | Exact source SHA256 | Exit | Wall | Result |
| --- | --- | ---: | ---: | --- |
| centered Taylor + complex rectangle Horner | b5ca02a0209d9eccd7cae8423f3fccfee4e9e15c76cf28a0d9f5d92686ed26c0 | 0 | 1.68 s | 216 eligible, 207 certified |
| independently generated integer Z₆(c), binomial translation + rational centered-ball estimate | 28e98c6d23646ed6e7494d857ee5b530a157720da90ec0af811bbd6442ec00a3 | 0 | 2.10 s | same 216, exact 207-cell agreement |

Commands: timeout -k 1s 35s python3 worker1.py > worker1.json; then the same for worker2.py. Worker 2 reads worker1.json only **after** independently computing its own list and intersects the lists. The frozen local run retained raw stdout, stderr, exit receipts and hashes in the separate dated research archive, not attached to this public commit.

**Controls:** c=0 gives z₆=0, rejected by both; the known center c₀ is certified by the first method. Every eligible cell passes exact annulus-inclusion arithmetic. Existing GitHub CI checks do **not** yet execute this new study. Local worker exits are not Lean verification or remote reproduction.
