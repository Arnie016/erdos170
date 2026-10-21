# Character obstruction for two first-denominator shifts at p=2521

**Predeclared claim contract:** For each prime p≡1(mod840), propose that c=3 or c=7 gives x=(p+c)/4 admitting positive x<y<z with 4/p=1/x+1/y+1/z. A single prime defeating both is the falsifier. Check primality, both impossibility arguments and a positive triple elsewhere. Search ascending primes p≤2,000,000, stop at first verified failure. Incomplete divisor enumeration is the strongest competing explanation.

## Exact lemma

Let n,c be positive, 4|(n+c), x=(n+c)/4 and A=nx. If 4/n=1/x+1/y+1/z for positive y,z, then

    (cy−A)(cz−A)=A²

because c/A=1/y+1/z. Both factors are positive: cy=A(1+y/z)>A. Thus there must be a positive divisor d of A² congruent to −A(mod c).

If gcd(A,c)=1, let χ be any multiplicative Dirichlet character modulo c. If χ(q)=1 for every prime q dividing A, but χ(−A)≠1, no solution y,z can exist. Every divisor d of A² would have χ(d)=1, contradicting d≡−A(mod c).

## Certified pair of failures

Take p=2521=1+3×840, prime by trial division through primes ≤50.

- For c=3: x=631 and A=2521×631. Both primes 2521 and 631 are 1(mod3). Every divisor of A² is 1(mod3), but d must be −A≡2(mod3). Impossible.
- For c=7: x=632=2³×79; A=2521×632. Its prime factors 2,79,2521 reduce to 2,2,1(mod7). All divisors belong to quadratic-residue subgroup {1,2,4}, while d must be −A≡5(mod7). Impossible.

This refutes only the c∈{3,7} first-denominator covering assertion, NOT the Erdős–Straus conjecture. An exact valid strictly ordered triple is

    4/2521 = 1/636 + 1/70588 + 1/5611746.

The independent local checker also exhausts x=631 through 635, making x=636 the least first denominator for this one prime. That minimality claim is a finite computational result; the obstruction at x=631 and 632 has the preceding ordinary proof.

Labels: COUNTEREXAMPLE (two-shift ansatz), ANALYTICAL_PROOF (general character criterion), PRIORITY_UNCHECKED (known fixed-shift literature exists), UNFORMALIZED. No novelty or global Erdős–Straus theorem claimed.
