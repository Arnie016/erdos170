# RUN REPORT · 2026-10-21

**Choice:** Compared #242 fixed-denominator template with #117 open draft PR #3, then selected #242 for a test with a genuine falsifier rather than repeating #117 finite calculations.

**Verdict:** COUNTEREXAMPLE to a proposed *two-shift ansatz*, not to Erdős–Straus. Prime 2521 (1 mod 840) fails shifts 3 and 7 by arithmetic character obstruction. Exact positive solution (636,70588,5611746) shows the full conjecture survives at that p.

**Execution contract:** Ascending primes p≡1(mod840), p≤2,000,000; stop at first verified failure of both templates. No random sampling or seed. Candidate verification by exact divisor products and identity 4xyz=p(xy+xz+yz), strictly x<y<z. Execution STOPPED at first eligible prime p=2521, so the entire upper domain was not scanned.

**Worker 1:** 35s hard timeout, process wall 1.76s, exit 0. sympy factors and integer divisor enumeration; SHA256 c7af59e1985c3ad57887b5668583af40b24ba14fa43a5e5442bd8e1ff85dbc68. **Worker 2:** separate 35s hard timeout, process wall 0.81s, exit 0; independent trial division, Cartesian powers of factors, exact rational identities; SHA256 800463e91af671eb1907c1f8a809f2fcbd315e02e5fe71a80774227e46ca0833. Both agree. Second finds first successful x=636 and rejects mutated z+1 triple. Full executed source and raw stdout are in the dated research-chat ZIP, not in this PR. Compact receipts EVIDENCE.json. No extra workers.

**Meaning:** This closes exactly one restricted covering assertion. General character lemma is proof-bearing. It does not create an infinite family, improve the 10^18 verification frontier, or settle #242. Fixed c=7 failures are already discussed in Dahan 2026. Literature priority unchecked.

**Outreach:** Gmail searched; no mathematics message sent. Publication is a draft proposal; repository CI for this study is NOT yet observed passing. Original sparse-ruler root files and schedules untouched.
