# Actual receipts, no simulated execution

Frozen case: c0=-3/4+i/2; radius 1/1000; j=0..6; negative control c0=0. No random draws. At most two sequential Python workers, each externally stopped at 35 seconds. ChatGPT Linux container, not the user's Mac/HPC.

1. `python3 worker1.py`: exit **0**, measured wall **1.1812 s**. Exact fractions. stdout PASS with strict squared norm inequality; SHA256 of executed source `0c4639bc43ddc87eb9525555b099bbce0483daf43014cab9ed3d358e1738d455`.
2. A separately written integer Gaussian numerator/denominator checker: exit **1**, measured wall **0.9598 s**. Its assertions reconstructed all center iterates, all error terms and corrupted-certificate rejection; it then crashed at final `print(...len(terms)...)` because `terms` was local to its validation function. A genuine process failure, not claimed as a clean pass. The original source SHA256 was `35eb109ff58ece9ee0c86e8d51e440e220d8128c07b7cdc2de9f58a98e3f5f50`. No third experiment/retry.

The SVG was rendered and visually inspected separately, with a uniform display transform to prevent circle/title overlap. It plots the exact-orbit coordinates at finite display precision, and enlarges the parameter disk illustratively; no full Mandelbrot background is simulated.

The original full local package includes both worker logs and the failed validator source; this review branch includes the successfully executed primary checker. **No dedicated GitHub CI/Lean validation of this new study is claimed.**
