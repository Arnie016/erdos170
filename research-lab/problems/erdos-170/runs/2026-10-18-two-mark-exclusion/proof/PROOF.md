# No complete two-mark replacement of a fixed length-500 near-ruler

**Validity:** EXACT_FINITE_CERTIFICATE; **literature priority:** PRIORITY_UNCHECKED; **formalization:** UNFORMALIZED. This is a local theorem, not an improved bound on the unrestricted minimum F(500) or on Erdős #170.

## Claim and dependency

Let A be the **exact 39 integers** in [candidate.json](../raw/candidate.json). A complete restricted ruler B in [0,500] has every positive distance from 1 to 500 as a difference of two marks. **Claim:** No complete 39-mark ruler can be obtained from A by replacing at most two of its marks with other integers in [0,500]. The earlier independently checked 2026-10-16 certificate rules out at most one replacement; it is not rerun here.

An explicit complete 39-mark set within two replacements of A would falsify this claim. The competing possibility, that coordinated two-mark changes escape the already proved one-change obstruction, is exactly what the finite calculation decides.

## Reduction to a complete exact domain

A contains both 0 and 500, has 37 interior marks, and misses exactly the distance 493. Any complete restricted ruler must contain both 0 and 500, because they are the only possible pair at distance 500. Thus successful two-mark replacements delete only interior marks.

The eight ways to express distance 493 with [0,500] are (j,j+493) for j=0,...,7. In A, the coordinates 0,...,6 and 500 are present, while 7,493,494,495,496,497,498,499 are missing. As A does not already represent 493, any repaired ruler covering 493 must insert **at least one mark** from T={7,493,...,499}. This remains true if both endpoints of the 493 pair are newly inserted.

The complement U=[0,500]\A contains 462 coordinates. Every genuinely two-mark completion therefore belongs to the exactly C(37,2)×(8×454+C(8,2))=**2,437,560** pair-replacement domain: delete two of 37 interior marks and insert two distinct elements of U, at least one in T. Deleting an endpoint cannot work. Re-adding a deleted coordinate effectively uses one or zero changes, already excluded.

## Independent exact checks

The first CPU implementation iterates all 666 deletion pairs and all eight distinguished insertions in T. For each intermediate 38-mark set it takes a missing distance d. A successful final inserted coordinate w must satisfy w=x±d for a retained coordinate x or for the first new coordinate. Checking all legal values w of this form is exhaustive. It tested 290,061 such w and found no complete ruler.

The second CPU implementation uses **no first-missing-distance pruning**. For every deleted pair, it reconstructs the retained-pair difference mask and separately calculates the bitmask gained by each of the 462 unused coordinates. It then loops over all 2,437,560 eligible unordered inserted-coordinate pairs, including the difference between the two insertions. It finds no case with all 500 required differences. This is independent algorithmic reconstruction using exact Python integers, not independent human review or a Lean proof.

Both runs finished successfully within a 32-second process timeout each on Python 3.13.5. The full source code and raw counts are in this directory. The 40-mark A union {499} is a positive control and not a 39-mark candidate.

## Boundary and next test

This exact certificate excludes two replacements from **this A only**. It cannot establish F(500)>39, the optimal finite ruler length, or a new asymptotic result. The smallest genuinely different local follow-up is a structurally filtered three-mark coordinated repair. A verified complete 39-mark ruler elsewhere would independently settle the saved local upper-bound goal without contradicting this certificate.

