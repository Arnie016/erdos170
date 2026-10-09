# All ordered two-coordinate alternating maps on F2^4

Status: exact finite certificate. Literature priority unchecked. No headline Erdős problem is claimed solved.

Alt(F2^4) has six scalar coefficients. Consequently all ordered pairs of scalar alternating forms comprise exactly 64^2=4,096 maps B:F2^4 x F2^4->F2^2. Different bases/presentations can yield the same commutation relation; the count is not an isomorphism-class count.

For each map, put an edge between nonzero vectors u,v when B(u,v)!=0. A color class consists of commuting vectors, whose span is isotropic by bilinearity. Its full preimage in a class-two realization is abelian. Conversely every abelian cover gives a coloring. Thus a equals chromatic number and omega equals clique number. Common radicals produce isolated vertices and independent twin classes; retaining them does not change these invariants.

The independently reproduced census is:

| omega=a | maps |
|---:|---:|
|1|1|
|3|105|
|5|1,050|
|6|1,260|
|9|1,680|

The two implementations use graph clique/coloring versus clique subset-DP and minimum covering by the 67 actual subspaces of F2^4. Both complete the full domain.

For a group with a=omega=n, the normalized ratio is n/2^(n/2). Its maximum over positive integers is 3/(2sqrt(2)): check n=1,2,3, then use (n+1)/n<sqrt(2) for n>=3. D8 realizes n=3, establishing that lower bound for a possible uniform constant. This family does not raise the benchmark.

This is not a census of all groups of order 64. The following day's kernel enumeration removes the two-output-coordinate restriction in dimension four.

Run `python3 src/worker1_census.py` then `python3 src/worker2_independent.py`.
