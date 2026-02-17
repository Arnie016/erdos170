# Erdős #170 Cycle 1 Research Report

## 1. Definitions

- `best_missing`: minimum uncovered-distance count found across attempts for a fixed `(N,m)` setting.
- `complete`: all distances `1..N` are represented, so `missing_count = 0`.
- `excess E`: `E = m - nint(sqrt(3N + 9/4))` where `nint` is round-half-up.

## 2. Deterministic Baseline Findings

- Sweep range: N=500..20000 step=100 (196 points).
- Ratio envelope m/sqrt(N): min=1.714643, max=1.788854, mean=1.740084.
- Completeness failures in deterministic constructor: 0.
- Fixed points:
  - N=500: m=40, base_term=39, E=1, m/sqrt(N)=1.788854, source=bot_prev_append_2, missing=0
  - N=700: m=47, base_term=46, E=1, m/sqrt(N)=1.776433, source=bot_prev_append_2, missing=0
  - N=1000: m=56, base_term=55, E=1, m/sqrt(N)=1.770875, source=bot_prev_append_2, missing=0

## 3. E-Hunt Outcomes by N

### N=500, m_try=39
- Best stage/candidate: `screen` / `delete_499_idx_38` (deleted mark 499).
- Best objective key: [0, 0, 0, 1, 243049]; missing=1, M25=0, M50=0.
- Coupled moves: attempts=8776, accepts=0; endpoint-hole fired=False.
- Status: no complete m-1 witness in this run budget.
- Best missing list: [493]
- Hole taxonomy: endpoint=0, mid_scaffold=0, other=1

### N=700, m_try=46
- Best stage/candidate: `screen` / `delete_697_idx_45` (deleted mark 697).
- Best objective key: [0, 0, 0, 3, 1432445]; missing=3, M25=0, M50=0.
- Coupled moves: attempts=8702, accepts=0; endpoint-hole fired=False.
- Status: no complete m-1 witness in this run budget.
- Best missing list: [690, 691, 692]
- Hole taxonomy: endpoint=0, mid_scaffold=0, other=3

### N=1000, m_try=55
- Best stage/candidate: `screen` / `delete_997_idx_54` (deleted mark 997).
- Best objective key: [0, 0, 0, 3, 2940302]; missing=3, M25=0, M50=0.
- Coupled moves: attempts=8712, accepts=0; endpoint-hole fired=False.
- Status: no complete m-1 witness in this run budget.
- Best missing list: [989, 990, 991]
- Hole taxonomy: endpoint=0, mid_scaffold=0, other=3

## 4. Summary

- No complete `m-1` witness was found in this cycle; results remain useful as reproducible near-miss data.
- Persistent-hole taxonomy is included above to drive next coupled-move targeting.
