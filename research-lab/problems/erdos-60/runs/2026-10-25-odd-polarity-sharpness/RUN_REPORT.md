# 2026-10-25 run receipts

**Predeclared question:** Is there a +1 improvement over the published 2q−4 local bound, for all two-added-absolute-edge and one-deleted-old-edge edits of an odd-q orthogonal polarity graph? Falsifier: any exact graph at 2q−4. Freeze acceptance before execution; at most two workers, sequential, <=35 seconds hard timeout each.

**Worker 1:** Independent incidence/path-based finite enumeration over declared q={3,5,7,11,13,17}, with early stopping on a falsifier. Stopped after q=3; all 15 unordered pairs of added conic edges, 150 relevant deletion checks. Found two C4 versus proposed three. A direct full-graph count checked the witness. Exit 0; measured wrapper wall 1.07s; script SHA256 462e2d170cb3d1545cb9b24dc720e9d16be6cd49d0bfebbd5cee89b976b3bbbd. Full first-worker input/output and script are retained in the dated companion archive, not claimed to have been rerun in CI.

**Analytical step:** A change of coordinates yielded explicit points a=[0:1:0], b=[1:0:−1], C_t=[1:t:t²], C_inf=[0:0:1]. The ordinary proof is for *all* odd prime powers, not merely tested primes. Source paper Lemma 2.8 is explicitly background.

**Worker 2:** A distinct graph-wide codegree C4 counter reconstructed the construction for complete declared q={3,5,7,11,13,17}. Expected H C4=0; H+one added C4=q−1; H+two added C4=2(q−1); H minus ab plus both adds C4=2q−4; e(H)=q(q+1)²/2. All passed. Final C4 counts respectively 2,6,10,18,22,30. Exit 0; wrapper wall 0.81s, program <0.14s. Source SHA256 3eee1dea0d9cd1cca292a5a93a8b103bd36d6e319492b07879ba8cc87ed360b4.

No paid API, extra computation, emails, device processes or schedule changes. Two local executions are distinct from GitHub CI, and the repository CI currently does not execute this new study. Exact local scripts, logs, claims and source-version hashes are available in the companion dated archive.
