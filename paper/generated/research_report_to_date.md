# Erdos #170 Research Report (To Date)

- Generated at (UTC): `2026-02-17T09:00:13.652547+00:00`
- Baseline summary: `results/excess_baseline/excess_summary.json`
- Cycle-1 summary: `results/e_hunt_delete_repair/summary.json`
- Cycle-2 root: `results/e_hunt_breakthrough_cycle2`

## 1) Problem And Definitions

- Goal in this finite campaign: find complete `m-1` witnesses at N=500,700,1000.
- `complete` means every distance `1..N` is represented (missing_count = 0).
- `best_missing` is the smallest uncovered-distance count observed for fixed `(N,m)`.
- Excess view: `E = m - nint(sqrt(3N + 9/4))`.

## 2) Deterministic Baseline (Wichmann-Oriented)

- Sweep: N=500..20000 step=100 (196 points).
- Ratio `m/sqrt(N)`: min=1.714643, mean=1.740084, max=1.788854.
- E distribution: {'-1': 1, '0': 61, '1': 134}
- Completeness failures in baseline sweep: 0.
- Fixed N=500: m=40, base_term=39, E=1, missing=0.
- Fixed N=700: m=47, base_term=46, E=1, missing=0.
- Fixed N=1000: m=56, base_term=55, E=1, missing=0.

## 3) Cycle-1 (Completed)

- Targets=3, complete hits=0.
- Success targets=[], global early stop=False.
- N=500, m_try=39: best_missing=1 at screen / delete_499_idx_38 (delete 499).
  missing_list=[493], M25=0, M50=0.
  holes(endpoint=0, mid_scaffold=0, other=1).
- N=700, m_try=46: best_missing=3 at screen / delete_697_idx_45 (delete 697).
  missing_list=[690, 691, 692], M25=0, M50=0.
  holes(endpoint=0, mid_scaffold=0, other=3).
- N=1000, m_try=55: best_missing=3 at screen / delete_997_idx_54 (delete 997).
  missing_list=[989, 990, 991], M25=0, M50=0.
  holes(endpoint=0, mid_scaffold=0, other=3).

## 4) Cycle-2 (In Progress Snapshot)

- `summary.json` present: False
- Seed artifact count (`best.json`): 458 / 456 (100.4%) (above config expectation due to resumed/extra candidate artifacts)
- Candidate artifact count (`candidate_summary.json`): 16 / 15 (106.7%) (above config expectation due to resumed/extra candidate artifacts)
- By target/stage (`best.json` counts):
  - N1000_m55: {'stage_screen': 36, 'stage_deep': 46}
  - N500_m39: {'stage_screen': 36, 'stage_deep': 72, 'stage_championship': 80}
  - N700_m46: {'stage_screen': 36, 'stage_deep': 72, 'stage_championship': 80}
- Best observed so far per target:
  - N=500: best_missing=1 (stage=screen, candidate=delete_499_idx_38, delete=499).
    missing_list=[493], M25=0, M50=0.
    coupled_accepts=0/36044, endpoint_window_accepts=0/25525, tail_trigger_count=57001.
  - N=700: best_missing=3 (stage=screen, candidate=delete_697_idx_45, delete=697).
    missing_list=[690, 691, 692], M25=0, M50=0.
    coupled_accepts=0/36342, endpoint_window_accepts=0/25768, tail_trigger_count=57001.
  - N=1000: best_missing=3 (stage=screen, candidate=delete_997_idx_54, delete=997).
    missing_list=[989, 990, 991], M25=0, M50=0.
    coupled_accepts=0/36469, endpoint_window_accepts=0/25789, tail_trigger_count=57001.

## 5) Research Interpretation

- Cycle-1 and current Cycle-2 confirm the same near-miss frontier:
  - N=500 misses [493] at m=39.
  - N=700 misses [690,691,692] at m=46.
  - N=1000 misses [989,990,991] at m=55.
- This is evidence of strong endpoint-tail rigidity in current move classes.
- `m-1` completeness is still open at these N in the current heuristic budget.
- For Erdos #170 relevance: this is finite-N structural evidence, not an asymptotic proof.

## 6) Reproducibility

- Baseline summary source: `results/excess_baseline/excess_summary.json`.
- Cycle-1 summary source: `results/e_hunt_delete_repair/summary.json`.
- Cycle-2 artifacts source: `results/e_hunt_breakthrough_cycle2/**`.
- Regenerate this report:
  - `export PYTHONPATH=src && python3 scripts/generate_research_report_to_date.py`
