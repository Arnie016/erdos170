# The minimum deficit after a one-mark reduction

Two independent programs exhaust all legal edits in the canonical ruler for each radius from 1 through 32. Each edit deletes two distinct original marks and inserts one fresh position in the original interval. The target is the number of positive distances in that interval that the result fails to measure.

| Radii | Deletion pairs | Legal edit triples | Minimum missing |
| --- | ---: | ---: | --- |
| 1–24, initial sample | 92,772 | 410,836,692 | 4 at radius 1, then radius + 5 |
| 25–32, held out | 121,164 | 1,246,657,308 | 30, 31, 32, 33, 34, 35, 36, 37 |
| 1–32, combined | 213,936 | 1,657,494,000 | All table columns agreed |

The [prediction](PREDICTION.md) was committed at `08cf4eb1692e7fbfb62722916efcf874e6b24134` before either implementation evaluated a held-out radius. At every radius from 3 through 32, the first minimizer in lexicographic coordinate order deletes both junctions and inserts $L+b$. At radius 2 this edit also attains the minimum of 7, while $(1,15,9)$ comes first. At radius 1 the junction edit misses 5 and the optimum is 4. The last three columns are the two deleted coordinates, in order, and the fresh insertion. [Full table](results.csv) · [Construction at all radii](JUNCTION-EDIT.md) · [Universal lower bound and limit](../DEFICIT-BOUNDS.md)

## How the calculations differ

The Swift host builds marks from the Wichmann gap word and indexes the original pairs by distance. The first Metal kernel asks for each deletion and distance whether any representing pair survives. The second assigns a thread to each deletion pair and insertion coordinate. For each lost distance $d$ it checks surviving partners at $z-d$ and $z+d$. It may stop after a candidate already exceeds an attained deficit. An atomic minimum with the coordinate as tiebreak preserves equal-deficit results in lexicographic order. The host recomputes the winning edit by enumerating all its actual distances.

The C++ implementation builds coordinates from five block formulas. For each deleted pair it reconstructs coverage from surviving pairs. For each missing distance it credits every fresh position reachable from a surviving mark, counting a restored distance only once per coordinate. It compares all fresh coordinates against their attainable gains and rechecks its winner from literal pairs. Both programs reject an incomplete baseline and verify a known possible repair and an impossible one. The Swift host separately checks a deliberately damaged coverage set. [Metal kernels](repair.metal) · [Swift host](RepairGPU.swift) · [Portable CPU verifier](verify_cpu.cpp)

The table also records the number of positive distances lost before insertion, summed over deletion pairs. Its unsigned 64-bit digest folds the descending lost distances and a zero separator after each deletion pair, using FNV multiplication modulo $2^{64}$. The digest is a cross-implementation comparison. The exact row minima are computed by the enumerators. The full CSV SHA-256 is `bcbd3e1c5693d5b1d292f9fda6326acd96ea1d7da7bc3ffbd2ab39b98e7de203`.

## Reproduce

Run from this directory on a Metal-capable Mac with the Swift compiler installed. Commands write generated binaries and trial output to a temporary directory.

```sh
trial_dir=$(mktemp -d)
xcrun swiftc -O RepairGPU.swift -o "$trial_dir/repair-gpu"
"$trial_dir/repair-gpu" repair.metal 32 > "$trial_dir/metal.csv"
diff -u results.csv "$trial_dir/metal.csv"
```

On either macOS or Linux with a C++17 compiler and Python 3, the following commands replay the table and independently check the exact missing sets for the proposed junction edit.

```sh
trial_dir=$(mktemp -d)
c++ -std=c++17 -O3 verify_cpu.cpp -o "$trial_dir/verify-cpu"
"$trial_dir/verify-cpu" 32 > "$trial_dir/cpu.csv"
diff -u results.csv "$trial_dir/cpu.csv"
python3 -B check_results.py
python3 -B check_template.py
```

The local run used an Apple M5 Max with 32 GPU cores and 36 GiB unified memory, macOS 26.6.2, Apple Swift 6.4.2 and Apple clang 21.0.0. Both compiled binaries ran concurrently for the full 32-radius confirmation. The measured wall times were 3.74 seconds for Metal with its Swift host and 6.83 seconds for C++. These are single-run receipts, excluding compilation. Linux CI runs the portable checker against the actual Mac CSV. Rebuilding on the GPU requires a Metal device.
