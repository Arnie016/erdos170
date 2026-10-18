# 2026-10-18 | Erdős #170 | Fixed near-ruler two-mark exclusion

**Result:** The 39-mark A in raw/candidate.json cannot be completed at length 500 by replacing at most two marks. Both exact algorithms found zero successes; the independent exhaustive method evaluated 2,437,560 eligible two-replacement sets. This is a **local** finite result, not a proof that every 39-mark ruler fails or a solution of Erdős #170.

- [Inspect the proof and exact finite reduction](proof/PROOF.md)
- [First implementation](src/two_move_search.py) and [independent exhaustive verifier](src/independent_exhaustive.py)
- [Raw search result](raw/search_result.json) and [independent result](raw/independent_result.json)
- [Execution receipt](RUN_REPORT.md) and [next checkpoint](CHECKPOINT.md)

Run from this directory:

```sh
timeout -k 2s 32s python3 src/two_move_search.py raw/candidate.json raw/search_result.json
timeout -k 2s 32s python3 src/independent_exhaustive.py raw/candidate.json raw/independent_result.json
```

**Literature:** PRIORITY_UNCHECKED. **Formalization:** UNFORMALIZED.

