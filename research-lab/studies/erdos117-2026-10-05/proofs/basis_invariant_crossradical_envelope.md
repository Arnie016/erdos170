# Basis-invariant cross-radical envelope

Status: analytical proof with exact finite matrix regression. Literature priority unchecked. This note does not solve Erdős #117.

Let B:V x V->W be alternating over F2, with dim W>=2. For nonzero lambda in W*, write beta_lambda=lambda o B. For independent lambda,mu define

`x(lambda,mu)=rank(beta_lambda | rad(beta_mu))`,

`Xi(B)=max_{lambda,mu independent} (x(lambda,mu)+1)(x(mu,lambda)+1)`.

An invertible change of central coordinates acts bijectively on W*, so Xi is basis-invariant. The maximum is only defined as written when dim W>=2; handle scalar or zero output separately.

## Clique bound

Put x=x(lambda,mu), y=x(mu,lambda). Choose U<=rad(beta_mu) nondegenerate for beta_lambda of dimension x, and U'<=rad(beta_lambda) nondegenerate for beta_mu of dimension y. Their intersection is zero: a vector in U intersect U' is orthogonal to U under beta_lambda, so nondegeneracy on U forces it to vanish.

An even-dimensional nondegenerate alternating space of dimension r over F2 contains r+1 vectors with all distinct pairings one. One construction realizes the (r+1)-by-(r+1) off-diagonal-ones Gram matrix, whose rank is r. Use {0} when r=0.

Take such cliques C in U and D in U'. Their sums are distinct. If c!=c', beta_lambda(c+d,c'+d')=1 because U' lies in its radical. If c=c' and d!=d', beta_mu detects the difference. Hence

`omega(B)>=(x+1)(y+1)`, and therefore `omega(B)>=Xi(B)`.

For a class-two group whose actual central quotient is F2^d, d>0, each quotient line lifts to an abelian subgroup. Thus `a(G)<=2^d-1` and `log2 a(G)<d`. It follows that

`omega(G)/2-log2 a(G)>Xi(B)/2-d`.

In particular, `log2 a(G)>=omega(G)/2-Delta` implies `Xi(B)<2(d+Delta)`. This may be a vacuous bound; no general near-extremality classification is inferred.

## Exact evaluation on one three-coordinate family

For beta0=sum x_i wedge y_i, beta1=u wedge w and beta2=sum_{i<=b} y_i wedge z_i with 1<=b<=a, encode scalar combinations by masks 1=beta0,2=beta1,4=beta2. Their radicals are:

| mask | radical |
|---|---|
|1|Z+<u,w>|
|2|X+Y+Z|
|3|Z|
|4|X+Y_tail+<u,w>|
|5|<x_i+z_i:i<=b>+<u,w>|
|6|X+Y_tail|
|7|<x_i+z_i:i<=b>|

For a source mask with bits c0,c1,c2, its rank on the radical of mask mu is: 2c1 for mu=1; 2a if c0=1, otherwise 2b if c2=1, otherwise 0 for mu=2; 0 for mu=3; 2(a-b)c0+2c1 for mu=4; 2c1 for mu=5; 2(a-b)c0 for mu=6; 0 for mu=7.

The maximum over the 21 unordered scalar pairs is `Xi=3(2a+1)`, attained by (beta0,beta1). Since d=2a+b+2, the resulting deficit is greater than a-b-1/2.

On the receiving slice Y_b+Z, the minimum restricted scalar rank is zero, so it is not a useful substitute for Xi. Conversely Xi can equal one when every nonzero scalar form is globally nondegenerate; induced radicals created by restriction are not detected. No universal hard-edge bound follows.

## Verification

`python3 src/verify_pencil_crossrank_envelope.py` recomputes all 21 scalar pairs for every 1<=b<=a<=16, totaling 136 parameter cases and 2,856 pair checks. It compares direct matrix restriction ranks with the table above.
