# Frontier Atlas

- Generated at (UTC): `2026-04-12T14:36:37.959618+00:00`
- Records analyzed: 6
- Cycle-1 source: `results/e_hunt_delete_repair/summary.json`
- Cycle-2 source: `results/e_hunt_breakthrough_cycle2/summary.json`

## 1) Core Pattern

The current frontier is not drifting toward new missing sets; it is collapsing into the same tail geometry.
Every observed certificate has a contiguous terminal run immediately before the endpoint, followed by one larger endpoint gap.
The furthest missing distance is always just below that endpoint gap, so the unresolved set sits in the last 10 or so distances rather than at the endpoint itself.

## 2) Per-Record Geometry

| source | N | m_try | best_missing | missing_list | furthest_missing_gap | terminal_run | endpoint_gap | run_prefix_gap | tail_density_25 |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |
| cycle1 | 500 | 39 | 1 | [493] | 7 | [486, 487, 488, 489, 490, 491, 492] | 8 | 14 | 8 |
| cycle1 | 700 | 46 | 3 | [690, 691, 692] | 8 | [682, 683, 684, 685, 686, 687, 688, 689] | 11 | 16 | 9 |
| cycle1 | 1000 | 55 | 3 | [989, 990, 991] | 9 | [980, 981, 982, 983, 984, 985, 986, 987, 988] | 12 | 18 | 10 |
| cycle2 | 500 | 39 | 1 | [493] | 7 | [486, 487, 488, 489, 490, 491, 492] | 8 | 14 | 8 |
| cycle2 | 700 | 46 | 3 | [690, 691, 692] | 8 | [682, 683, 684, 685, 686, 687, 688, 689] | 11 | 16 | 9 |
| cycle2 | 1000 | 55 | 3 | [989, 990, 991] | 9 | [980, 981, 982, 983, 984, 985, 986, 987, 988] | 12 | 18 | 10 |

## 3) Shared Motifs

- N=500: shared_missing=True, shared_terminal_run=True, furthest_missing_gaps=[7].
  missing_lists=[493].
  terminal_runs=[486, 487, 488, 489, 490, 491, 492].
- N=700: shared_missing=True, shared_terminal_run=True, furthest_missing_gaps=[8].
  missing_lists=[690, 691, 692].
  terminal_runs=[682, 683, 684, 685, 686, 687, 688, 689].
- N=1000: shared_missing=True, shared_terminal_run=True, furthest_missing_gaps=[9].
  missing_lists=[989, 990, 991].
  terminal_runs=[980, 981, 982, 983, 984, 985, 986, 987, 988].

## 4) Geometric Read

The most useful interpretation is scaffold rigidity: the tail block is already dense and interval-like, so the search is probably fighting the last coarse gap rather than isolated endpoints.
That points toward multi-mark tail surgery, residue-preserving block moves, or a local exact repair model instead of more single-distance targeting.
