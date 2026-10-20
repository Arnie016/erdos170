# Exact 39-mark length-500 complete ruler

**EXACT_FINITE_CERTIFICATE; PRIORITY_UNCHECKED; UNFORMALIZED.**

For N=500, the 39 integers

```
0, 1, 2, 3, 4, 5, 11, 22, 33, 44, 55, 66, 89, 112, 135, 158, 181, 204, 227, 250, 262, 285, 308, 331, 354, 377, 400, 423, 435, 447, 459, 471, 483, 495, 496, 497, 498, 499, 500
```

form a complete restricted ruler: for each d=1,...,500, two displayed integers satisfy b-a=d, a<b. The successive 38 gaps are `1^5, 6, 11^5, 23^8, 12, 23^7, 12^6, 1^5`, where the exponent means repetitions. This is a shortened Wichmann-pattern ruler (r=5,s=16) with one 23-gap reduced to 12, yielding total length 500.

**Inspectable proof:** `witnesses.csv` has 500 rows, one labelled d=1..500; each row supplies a,b in the displayed set with b-a=d, with arithmetic over integers. The independent `independent_verify.py` checks all positive differences using a bitset and separately rebuilds those 500 witnesses by explicit pair enumeration. This is a finite certificate, not a heuristic coverage percentage.

The prior project 39-mark near-ruler missed distance 493; adding mark 499 gave a verified 40-mark ruler. A separate local two-move impossibility certificate applies ONLY to that earlier candidate. A different 39-mark solution does not contradict it. The positive 40-mark control and the negative 38-mark delete-262 control pass. `F(500)<=39` follows, but not `F(500)=39`; no 38-mark exclusion is claimed. The asymptotic Erdős #170 limit remains unrelated to this finite construction. The Wichmann family is classical; publication priority or novelty of this 39-mark instance has not been established.
