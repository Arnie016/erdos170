# A boundary between a proof and a proof method

![Nine mathematical parent squares, each split into sixteen colored children. Teal marks previously certified whole-square exterior; blue marks an exterior center whose square remains unresolved; gold marks a center failing to escape by iteration six.](CENTER_AUDIT.svg)

**New finite diagnostic:** 20 of the 42 remaining centers escape by iteration six; 22 do not. The 22 associated **whole squares** cannot satisfy a uniform sixth-iterate escape criterion. This is not evidence those points belong to the Mandelbrot set and it does **not** enlarge the previous certified exterior area.

Read [PROOF.md](PROOF.md) for the exact scoped theorem. Reproduction: run `timeout -k 1s 35s python3 worker1.py > worker1.json` followed by `timeout -k 1s 35s python3 worker2.py > worker2.json`, in this folder. The second checker independently reconstructs all 42 exact fractions and compares them with the first. The verified local run exited 0 twice, but new-study GitHub CI has not been observed. The preserved full raw outputs and execution receipts are in the dated downloadable local research package.

**Status:** exact finite point classification, known proof method, priority unchecked, Lean unformalized. No published full-cell certificate for today's blue or gold tiles.
