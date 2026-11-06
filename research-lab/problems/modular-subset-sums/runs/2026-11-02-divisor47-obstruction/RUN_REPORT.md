# Execution receipts | 2026-11-02

Frozen computational domain: d=11,47 and target polynomial P=X^8+X^5+X^4+X^3+1. Controls: X^2+1 giving resultant 1 for odd d and Phi_11 giving zero resultant for d=11. No randomness. Claim acceptance: independent signed integer resultants and all controls match.

- Worker 1: timeout -s KILL 35s python3 worker1.py worker1.json; exit 0; external wall 1.188787333 s; internal 0.000416102 s; local SHA256 a8b6f28df1e48dc2b6468943be7be3b835d28fec7d46a0b1fbd52ffcb6764779.
- Worker 2: timeout -s KILL 35s python3 worker2.py worker2.json; exit 0; external wall 0.808130743 s; internal 0.010538501 s; local SHA256 6a2d5780f40072be133cf514f5fe8d0729b166fddda544856a6c742270f53af1.

Output: Res(F_11,P)=1 and Res(F_47,P)=1,274,641=1129^2. All controls pass. Two workers total, no restarts.

Analytical strengthening after acceptance: The frozen claim contract initially considered only odd n and the units-only norm. The previously derived general odd-N identity (independent of unit status) yields the stronger all-parity progression n≡9 mod23. This does not alter any finite-domain acceptance criterion. No native Lean verification.

Research context, GitHub source/paper automation and all schedules unchanged. No email sent. New-study GitHub CI reproduction not yet confirmed.
