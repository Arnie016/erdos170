# October 15, 2026: exact independent checkers for the 255 fixed-plane cases

**Claim status:** EXACT_FINITE_CERTIFICATE, local execution. Literature priority: PRIORITY_UNCHECKED. Formal status: UNFORMALIZED. GitHub-CI reproduction of this particular 255-case result: PENDING.

This supplements [STUDY.md](STUDY.md) without broadening its claim. Let V=F_2^5; the 10 scalar-form coefficients are ordered 01,02,03,04,12,13,14,23,24,34. Fix the two-dimensional form plane P=span(129,66), where 129=e01+e23 and 66=e02+e14. The finite domain is exactly the 255 different three-dimensional scalar spans containing P, with each third form selected canonically modulo P. These are not asserted to represent 255 different isomorphism classes.

**Falsifier:** a canonical third form C for which chi(Gamma_C)>omega(Gamma_C), or one invalid lower/upper witness. The plausible competing explanation is that fixing P already forces relatively large cliques, concealing gaps in other orbits.

Two independently implemented C++17 checkers were run sequentially, each once, under a 30-second hard timeout with a two-second kill grace. Both exited 0. Clique worker wall time 0.03 s; cover worker wall time 0.01 s. The first evaluates the alternating form through exterior-square masks and enumerates exact maximum cliques. The second reconstructs the three 5-by-5 alternating matrices independently, checks every clique mask, enumerates all 374 vector subspaces of F_2^5 and constructs matching totally isotropic covers. A size-k clique gives chi>=k, and an isotropic k-subspace cover gives chi<=k. Thus both bounds match for each of the 255 cases.

**Histogram:** omega=chi=9 for 4 scalar spans, 11 for 6, 13 for 69, 14 for 112, 15 for 40, 17 for 24. This agrees with the earlier local-only report but is a new checker implementation, not a byte-for-byte reproduction of the historical checker.

**Reproduction:** From this directory compile and run:

~~~sh
c++ -std=c++17 -O2 -Wall -Wextra src/check_255_cliques.cpp -o clique
c++ -std=c++17 -O2 -Wall -Wextra src/check_255_covers.cpp -o covers
timeout --signal=TERM --kill-after=2s 30s ./clique > cliques.csv
timeout --signal=TERM --kill-after=2s 30s ./covers cliques.csv covers.csv
~~~

Columns of the raw certificates are C,omega,clique_mask and C,omega,cover_count,subspace_masks (the last with colon-separated isotropic-subspace point bitmasks). Every instance must have a clique and a cover with matching k. Worker 2 checks isotropy and full 31-point coverage, not only a solver flag. The complete executed raw CSVs, actual worker logs and separate ordinary proof remain in the dated 2026-10-15 task-chat ZIP, not yet in this branch.

Local SHA256 of cliques.csv: 3604cffe690fc0b1e69c9415ed67ec95bf987458e6576ec27870ee62b3ea9957

Local SHA256 of covers.csv: 1c3d7814be3774e80299788d7e3943532f9bcc91e0b765c5dd596aae1cb09d5e

**Scope boundary:** The ordinary cocycle construction realizes these forms as groups of order 256 with Z(G)=G'=F_2^3 and G/Z(G)=F_2^5. This establishes a(G)=omega(G) in the specified family, not the universal three-form case and not a new bound on h(n) over all groups. No Lean claim. The external literature priority is unchecked.

**Next test:** choose a faithful two-form plane outside the GL(5,2)-orbit of P with baseline omega<9, then independently certify its full extension slice. A verified chi>omega would refute universal equality for the broader three-form family.

Public historical source: Guillaume Lecomte, arXiv:2608.20507v1 (20 Aug 2026), https://arxiv.org/abs/2608.20507. No unpublished work or private correspondence is used here.
