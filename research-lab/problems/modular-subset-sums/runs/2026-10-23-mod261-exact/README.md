# Modulo 261: all eight-number cases classified

Read [PROOF.md](PROOF.md), [CHECKPOINT.md](CHECKPOINT.md) and [RUN_REPORT.md](RUN_REPORT.md). Source contains two independent exact recurrences. This is a finite certificate, not a Lean proof or all-n solution. The three perturbation templates are known published constructions; finite-completeness priority is unchecked.

From this directory:

```sh
mkdir -p results
# Run sequentially; each limit includes compilation.
timeout --signal=KILL 35s bash -c 'g++ -std=c++20 -O3 src/search_a.cpp -o /tmp/mod261-a && exec /tmp/mod261-a results'
timeout --signal=KILL 35s bash -c 'g++ -std=c++20 -O3 src/check_b.cpp -o /tmp/mod261-b && exec /tmp/mod261-b results'
```

Expected complete outputs:22 normalized leaves,252 sign classes,64512 admissible sets,three signed-unit orbits and maximum dyadic overlaps7 or8. All252 canonical sets are regenerated in results/full_sign_normalized_sets.txt and retained as a CI artifact. Historical local results are not substituted for a failing fresh execution. No install or paid service required.

[Reusable external results and tools](REUSABLE_REPOS.md) are a separate research guide, not part of the proof.
