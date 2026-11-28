# Match log

The opening commit fixes the two targets, prior results, extensions, and scoring. Submission and referee records belong in the match replay. New rounds can reuse the rules and must record their own baselines before play.

## November 28 outcome

The group desk earned 13 and the ruler desk 12 under the frozen rules. The referee challenged isolated vertices in the group construction and requested the exact finite boundary for the ruler. Both players answered with checked evidence. The full binary map theorem earned the deciding Lean point. [Scored replay](../README.md)

After scoring, the M5 Max search found a pattern in the number of missing ruler distances. The first 24 radii were exploratory. Commit `08cf4eb1692e7fbfb62722916efcf874e6b24134` froze the next eight predictions. Independent C++ reproduced every table cell, including the exact optima and coordinates. A written proof now supplies the proposed upper bound for every radius and a uniform lower bound differing by at most 13. The limit of the best deficit divided by r is one. [Proof](../../../problems/erdos-170/runs/2026-11-28-exterior-repair/DEFICIT-BOUNDS.md)

The full source revision `a54a4b5ce6061f2c9375bd33641f4f7ae4284a21` passed all four hosted jobs, including a fresh Lean build, all 27 dependency audits and all eight earlier studies. An initial local replay of the October studies selected Apple's default compiler for a historical GNU-only header. Replaying those programs with installed GNU C++ 15 passed. The November portable verifier used Apple clang locally and GNU C++ on the hosted runner. [Source CI](https://github.com/Arnie016/erdos170/actions/runs/46112990968) · [Public pull request](https://github.com/Arnie016/erdos170/pull/11)

The run budget was 45–60 minutes and an estimated 25,000–60,000 model tokens across the three agents and lead. The source and first hosted verification finished within the time budget. The runtime did not surface actual token usage or dollar cost. External compute purchases and paid API spend were $0. An out-of-project Lake version probe populated a separate Lean 4.36.0 cache on the research Mac. All scientific builds used the existing Lean 4.19.0 project pin.

The outstanding mathematical test is an edit missing at most r+4 distances for some radius greater than 32, or a universal matching lower bound r+5. Literature priority and independent human mathematical review remain open. Human comprehension review of the board is `NOT_TESTED`.
