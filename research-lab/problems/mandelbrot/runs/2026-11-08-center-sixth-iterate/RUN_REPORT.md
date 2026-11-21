# Local execution receipt | 8 November 2026

**Frozen domain:** exactly 42 zero-bit child centers drawn from the nine parents in PR #8 at c6bfd7363fabe6ba52dd7c28571eeba380d33c42. No random seed. **Acceptance:** two independent exact constructions return identical reduced norm² fractions for every center; classify strict >4, strict <4 and equality separately. Controls: z₆(0)=0 and |z₆(c₀)|>2.

| Worker | Command (run sequentially) | Exit | Measured wall | Result |
|---|---|---:|---:|---|
| Fraction-pair | `/usr/bin/time -f 'elapsed_seconds=%e cpu_seconds=%U exit=%x' timeout -k 1s 35s python3 worker1.py >worker1.json 2>worker1.stderr` | 0 | 1.09 s | 42 centers, 20 >4, 22 <4 |
| Homogeneous integers | `/usr/bin/time -f 'elapsed_seconds=%e cpu_seconds=%U exit=%x' timeout -k 1s 35s python3 worker2.py >worker2.json 2>worker2.stderr` | 0 | 0.69 s | exact 42/42 agreement, no equals |

Original local executable source SHA256: worker1 `ad3858494efa7d696e2f91e72b8ca60a28e2742c5ea4cc99e6ec1185245d216d`; worker2 `80abd23d305b5eb30bac093ff58eb353ab2ca8adea39c588c0a3a19615359bbe`. Local archive includes full raw JSON plus stderr/time receipts; source equality with this committed version must be checked before labeling this GitHub code reproduced.

Exactly TWO sequential workers, 35-second limits, no retries. No GPU/paid/remote computation. The prior 102 accepted cells were not reexecuted. New-study GitHub CI and Lean verification not claimed.
