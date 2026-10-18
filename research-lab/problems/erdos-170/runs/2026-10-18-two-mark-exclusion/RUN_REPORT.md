# Exact run receipt | 2026-10-18

Source candidate: public Arnie016/erdos170 commit 8dd05629581e3cb96b98c099f1e95c680d2d05ff, results/e_hunt_delete_repair/summary.json; previous one-replacement proof in the dated 2026-10-16 research package (not replayed today).

Both current workers were run sequentially with hard command timeout "timeout -k 2s 32s", kill/wait cleanup and an enclosing 35-second process cap, Python 3.13.5, CPU exact integers. No GPU or paid service.

1. two_move_search.py: exit 0, inner elapsed 7.272905 sec; 666 deletion pairs, 5,328 distinguished-first-insert trials, 290,061 logically sufficient second-insert checks; no completion.
2. independent_exhaustive.py: exit 0, inner elapsed 7.281738 sec; **all 2,437,560** eligible two-insert final sets tested; zero complete sets. Baseline missing {493}; A union {499} complete with 40 marks.

SHA-256 original locally executed files:
- two_move_search.py: 7856db375138910a12738f11ea2d56079341cad3b775164e85c551f2fe23e3d1
- independent_exhaustive.py: 25d0f37f2e27085d688e3694e95ea0c98da72635cb215bbc08a8fe9453cb8ff9
- candidate.json: 01349857068ec28abb43f8541acce110d7b3f4fcde2467b14a95ee4d1a9c6f33

Publication-source hashes must be checked against these before promoting exact reproduction. This is a finite local certificate only, neither a global theorem nor a Lean proof.

