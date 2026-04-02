# Rigidity Memo

- Generated at (UTC): `2026-04-02T17:23:18.603521+00:00`
- Root: `/Users/arnav/.codex/worktrees/a238/erdos/results/e_hunt_breakthrough_cycle3_tailfocus`

## What The Current Data Says

- N=500, m_try=39:
  - best row: `deep` / `delete_499_idx_38` (delete 499)
  - missing=[493]; M25=0; M50=0
  - hole taxonomy: endpoint=0, mid_scaffold=0, other=1
  - move metrics: coupled=0/285261, endpoint_window=0/253865, tail_triggers=298801, endpoint_injection=False

## Rigidity Read

- The search is currently concentrated on a single frontier instance, so the main risk is overfitting one move class rather than expanding the evidence base.
- The best next step is to shard the same frontier across disjoint move ablations and seed bases, while keeping the current target live as the mainline run.

## Recommended Next Step

- Refresh this memo after any new candidate summary, then publish the deltas to GitHub.
- If the frontier is unchanged, do not rewrite the memo; if it changed, keep only the smallest changed artifact set plus the generated summary JSON.
