# Reproduction and trust model

## Full computational replay

Run `python3 research-lab/tools/verify_all.py` with Python 3.10+ and g++ supporting C++17. No external Python package is required. The driver copies the 13 study programs into a new directory under `research-lab/_generated/`, then runs them sequentially with two C++ compilations. Each individual process has a 35-second hard walltime limit with process-group cleanup on timeout.

The driver requires the complete expected domains and exact outputs, not simply an exit code. Examples: all 2,825 kernels with identical clique/cover records; all 174,251 scalar planes with the expected histogram and independent covers; exactly 66 canonical modular sets and all 4,224 sign lifts. Large JSONL censuses are regenerated. CI uploads logs, raw records, source fingerprints and `SUMMARY.json` as an artifact of the exact commit.

The build/replay is a publication check, not a daily exploration allowance. Future daily research still uses its separately authorized resource budget.

## Lean layer

```
cd research-lab/lean
lake build
python3 -B audit_axioms.py
```

Pinned toolchain: `leanprover/lean4:v4.19.0`. Standard library only. The twelve initial declarations use ordinary kernel-reduced `decide` or explicit proof terms, not `native_decide` and not admitted proofs. The printed axiom report from the first successful build contains ten declarations without axioms and two cover checks using only `propext`.

These declarations verify concrete finite witness predicates and a product noncommutation implication. They do not yet formalize the abstract group reduction, every proof note or the large censuses. The C++ and Python computations remain a separate trusted executable layer.

## November duel

The replay reducer verifies the recorded points, feedback links and stage ordering. CI also reproduces the entire 32-radius Metal table with portable C++, checks both independent odd-field examples and runs the exact finite boundary for the all-pairs ruler obstruction. [Match commands](benchmarks/erdos-duel/README.md) · [Metal and CPU commands](problems/erdos-170/runs/2026-11-28-exterior-repair/gpu/README.md)

On macOS the October regression driver invokes `g++` by name and its historical sources use GNU headers. Select an installed GNU C++ toolchain for that driver. The November minimum-deficit checker builds with Apple clang or GNU C++.

## Mathematical checking

Each study states its domain and what evidence establishes. Lower clique witnesses and upper subgroup covers have opposite inequality directions; matching values certify equality. The broad low-dimensional conclusions require the ordinary group-to-bilinear reduction in addition to the finite programs. Several historical notes have separate asymptotic hypotheses; their finite regression does not establish those hypotheses.

The paired implementations are distinct algorithms, not independent human reviewers. Some elementary helpers (such as bitwise parity) necessarily express the same mathematical operation. No Lean success should be read as proving the entire repository.

## Import boundary

The historical import covers the October 3–9 #117 studies and the modular n=6 study. The November runs have their own source and receipts under the problem directories. Scripts are ported to relative output paths and public notes are edited; they are not advertised as byte-for-byte archival copies. Original private archives, correspondence, local personal paths and binaries are excluded. Source modifications require a fresh CI run. Original #170 files remain untouched outside this subtree.

## Known setup failure

The first Lean workflow stopped before compilation because `lake-manifest.json` was absent. A dependency-free manifest fixed this. Run 37905339322 then built the declarations and passed the axiom check. This was a setup failure, not a falsified mathematical theorem.
