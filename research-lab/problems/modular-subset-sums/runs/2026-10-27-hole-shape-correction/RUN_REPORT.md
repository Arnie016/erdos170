# Run report | 2026-10-27

Selected the prior modular t=5 n=9 centered-hole compatibility target over the older #170 local-repair tactic because it offered a structural falsifier without repeating finite censuses.

**Predeclared target:** N=517, base H={0,±1,±3}; under translation, the existence of one- or two-generator tilings is invariant (though the base H contains 0 and therefore cannot itself be the actual holes of a complete subset-sum system). Check all 460 invertible first generators, construct exact two-shift compatibility edges and calculate the graph's maximum clique; a genuine nine-generator realization would also require the anchored empty subset, not just a clique.

**Actual bounded execution:** first worker attempt could not save outputs because of a directory permission error; count it as a consumed worker and certify no numerical output from that attempt. One final worker process executed two distinct deterministic implementations sequentially under a 35-second hard timeout, with exit 0 and external wall 0.88s. They agreed on 156 first-level candidates, 308 unordered compatible pairs, maximum clique 9 and witness {2,3,4,8,16,32,64,128,256}. Independent algorithms solved cyclic hole-gap equations versus direct binary recurrences and checked each four-block equation. No paid resources, GPU, Mac execution or random seed.

**Finding:** the candidate hole shape is already realizable through a known template for *all* n≥3, so the supposed non-template target was incorrect. An inspectable ordinary proof, independent of computation, is published as PROOF.md. The locally retained raw finite check and code have not been ported to public CI; no new-study CI reproduction or Lean verification is claimed.

**Gmail:** one strictly math-focused recent-reply search; no outgoing message. **New result label:** CORRECTION, not a novel global theorem. No generated catalog registration because this is a clarification of an already-known construction, not a new exact result.
