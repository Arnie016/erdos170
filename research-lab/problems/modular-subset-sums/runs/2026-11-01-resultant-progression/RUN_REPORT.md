# Execution receipts | 2026-11-01

Chosen one modular distinct subset-sums question: is Res(F_517,P4) exactly 1 or not, and can that answer rule out the centered radius-(1,4) holes along a nontrivial n progression? Before execution the acceptance condition was frozen: two independent computations must agree on each of three polynomials; P3 positive control gives 1 and Phi11 zero-resultant control gives 0. The second competing candidate was a global #117 central-index constant, not sufficiently scoped for a bounded proof in this unit.

Complete finite domain: N=517 and three fixed polynomials P4=X^8+X^5+X^4+X^3+1, P3=X^6+X^4+X^3+X^2+1, Phi11=1+...+X^10. No random sampling or seed.

Worker 1, Linux CPython, exact companion matrix power and fraction-free determinant: command 'timeout -s KILL 35s python3 worker1.py worker1.json'; exit 0; external wall 1.15s; source SHA256 e153fb00325fb4465925893d82152591cb49c034bd1c67d14db4b662e555cdf0.

Worker 2, Linux CPython, independent polynomial modular power and Sylvester determinant: command 'timeout -s KILL 35s python3 worker2.py worker2.json'; exit 0; external wall 0.77s; source SHA256 96fa8b8b49b3fc80dbb56d615a6e33be37c302268c6a28f9f937363996cf59a7.

Both agreed exactly: target norm 4493599431468313730139668192986121524839260273570500546030600490466361; positive control 1; zero control 0. Files worker1.json and worker2.json contain the observed raw outputs. Their determinant implementations both use the standard Bareiss identity; their matrix constructions and exponentiation are independent. The infinite congruence progression is derived by ordinary exact arithmetic, not numerical extrapolation.

No more than two experimental processes, no timed-out restart, no GPUs or paid compute. No new-study Lean proof. No mathematical email sent or unrelated inbox read. GitHub CI may run existing verification jobs on PR update, but no new-study CI acceptance is claimed in this receipt. Main source/paper automation untouched.