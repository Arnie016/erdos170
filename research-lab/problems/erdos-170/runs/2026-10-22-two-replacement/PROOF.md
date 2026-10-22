# Exact two-replacement exclusion for one 39-mark ruler, N=500

**Statement (EXACT_FINITE_CERTIFICATE, PRIORITY_UNCHECKED, UNFORMALIZED).**
Let A be the 39 positions listed in candidate.json. No 39-element B contained in {0,...,500} and obtained by replacing at most two marks of A realizes all positive distances 1,...,500. Among sets with *exactly* two replacements, the minimum number of missing distances is 2.

**Analytical complete-domain reduction.**
Distance 500 requires 0 and 500 as marks, so these endpoints cannot be removed from any complete candidate. Every representation of the missing baseline distance 493 has the form (k,k+493), 0 <= k <= 7. The original A contains 0,...,6 and 500 but contains none of 7 or 493,...,499. Any repaired complete B must therefore add at least one of S={7,493,494,495,496,497,498,499}. There are choose(37,2)=666 ways to remove two interior original marks. There are 462 unused original positions. The unordered new-position pairs meeting S are 8*(462-8)+choose(8,2)=3660. Therefore 666*3660=2,437,560 unordered 2-replacement sets include *every possible complete 2-replacement candidate*. The 0- and 1-replacement domains are separately accounted for.

**Exact independent algorithms.**
`src/worker_a.cpp` uses 64-bit difference masks and checks each distance bit. `src/worker_b.cpp` uses pair-difference counts for the 37 retained original marks, then checks the missing distances by exact marked-position membership after the two additions. Both cover the entire 2,437,560-element domain with different predicates and independently output 0 complete, minimum 2 missing. Worker A additionally checks all 37*462=17,094 one-mark neighbours with zero complete; A itself misses precisely 493. Inputs are literal hardcoded positions in both workers, mirrored in candidate.json.

**Scope.**
This is a certified obstruction to *one particular local edit ball*, not a proof that F(500)>39, no ruling out three replacements, and no improvement to the asymptotic Erdős #170 bounds. Neither checker is Lean-verified. The prior October 16 one-mark audit is a baseline, not the new discovery. Executed Linux receipts are in RUN_REPORT.md; new-commit CI reproduction is pending.
