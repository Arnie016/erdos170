# Daily mathematics portfolio, 2026-10-22

**Selected #170**, comparing the recovered finite two-swap question against a new #242 fixed-denominator ansatz direction. The 2026 Dahan preprint already gives stronger fixed-shift obstacles, whereas the #170 edit-ball gap was clearly defined and entirely checkable. This is a genuine second-radius test, not a rerun of the October 16 one-edit audit.

**Verdict:** EXACT_FINITE_CERTIFICATE, PRIORITY_UNCHECKED, UNFORMALIZED. No complete candidate in the entire 2,437,560 exactly-two-edit domain. Two independent C++20 checkers agree on 2,437,560 tested, 0 complete, and minimum missing distance count 2. The baseline A misses 493. Worker A additionally checked 17,094 one-edit neighbours, 0 complete.

**Executed receipts:** gcc g++ 14.2.0, C++20, -O3. Each worker executed separately with hard timeout `/usr/bin/timeout --signal=TERM --kill-after=2s 35s`; both returned exit code 0 without timeout. Worker A measured wall 0.806467928 s; worker B 0.133943824 s. Raw outputs saved in the results directory. No RNG or GPU, paid service, Mac process or autonomous subagents. Compilation completed separately before the two experimental workers. C++ warnings involved boolean operator parentheses, not correctness.

**Audit limits:** The workers hardcode the input literal and do not parse candidate.json; verify both when changing inputs. These exact integer enumerations are not a formal Lean proof or independent human audit. The experiment excludes only a two-substitution neighborhood of the specified 39 marks, not the existence of any 39-mark ruler and not the #170 asymptotic limit. Reviewable PR submission does not equal CI verification, and this result is deliberately **not added to generated catalog navigation until a new-study workflow receipt meets its evidence_run requirement**.

**Outreach:** none. Existing PR #2 is an unrelated older root enhancement, PR #3 is a #117 fixed-plane study. No private material included.
