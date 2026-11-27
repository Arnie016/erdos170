# Exact execution receipts | 5 November 2026

**Frozen domain:** c0=-3/4+i/2, iterations 0..6, radii {1/1000,1/750,1/500,1/400,1/250,1/100}, no sampling/seed. Negative control: z6(0)=0 cannot be an escape witness. Two sequential worker processes in an ordinary Linux sandbox, not the user's Mac/HPC. Each invoked under timeout -k 1s 35s. Neither timed out or was restarted.

| Worker | Source SHA256 | Exit | Measured wall | Result |
|---|---|---:|---:|---|
| worker1.py, exact Taylor delta coefficients plus upward square-root bounds | 05f8ac85a5b0055a87c8de100accabc6f517cccb8df040e0ad3c98b95caa2d12 | 0 | 1.15 s | All 33 coefficients, radius 1/250 certified |
| worker2.py, integer polynomial plus binomial shift and rational L1 | 48b36972c5ccaa91ff31091491aa72132b6b6b747abb09115243458a3c63df0e | 0 | 0.76 s | Same coefficients, radius 1/250 certified |

Commands were: timeout -k 1s 35s python3 worker1.py and timeout -k 1s 35s python3 worker2.py. Both also rejected the c0=0 escape claim. Both fail to certify radius 1/100, an informative bound failure, not a counterexample to exclusion. Local full stdout JSON and command receipts are in the dated package linked in the task conversation, not automatically uploaded to GitHub. Re-run each source and inspect the exact boolean for radius 1/250, not merely the program's exit code. The checkers' base-level assertions alone do not fail the program when a larger radius fails certification.

Source hashes refer to *locally executed exact versions*. New-study remote CI and Lean checks have not been observed. Existing historical CI success is not evidence for this result.

