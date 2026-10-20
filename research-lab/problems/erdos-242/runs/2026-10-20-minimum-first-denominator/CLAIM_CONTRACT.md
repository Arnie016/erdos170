# Claim contract | P242-FIRST-20261020

Exact statement: For every positive integer t divisible by 7 whose prime divisors are 1 modulo 3, with n=t², the minimum first denominator D(n) among strictly ordered positive triples 4/n=1/x+1/y+1/z is (n+7)/4.

Definitions: D(n) is the least x with x<y<z and positive integral denominators. The smallest a priori x is x0=(n+3)/4 because n≡1 modulo 24.

Dependencies: elementary modular arithmetic, multiplicative order modulo prime factors of k²+k+1, factorization (3y−m)(3z−m)=m², positive integer construction.

Strongest competing explanation: a smaller x=x0 might work despite examples apparently excluding it. Existing literature may already imply the proposed exact minimum.

Explicit falsifier: an allowed t with x=x0 giving valid positive denominators, or an allowed t where the constructed x1 triple fails.

How either outcome matters: a proof settles one infinite exact-denominator subclass; a counterexample invalidates the proposed obstruction. Neither resolves the global conjecture.

Acceptance: auditable ordinary proof for all quantified t plus bounded exact-integer regression with a negative control. Stop after acceptance or correction. Novelty and Lean formalization are separate.
