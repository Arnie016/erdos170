# All symmetric scalar three-bin systems lose to a rational two-kernel system

**Claim:** SIDON-3BIN-FULLSCALAR-2026-11-20. **Validity:** ANALYTICAL_PROOF, with an auxiliary EXACT_FINITE_CERTIFICATE on a stated 190-pair grid. **Literature:** known method; specialization priority unchecked. **Formal:** unformalized.

## Frozen statement and definition

Apply equations (2)–(4) and Theorem 2.1 of Hou–Zhao, arXiv:2607.01169v3 (4 September 2026). Fix m=3, L=1, R=1, one symmetric probability kernel p=(t,1−2t,t), 0≤t≤1/2, and arbitrary real boundary vector w=(w0,w1,w2), extended by wj=1 for j≥3. Require all four covering inequalities Cq≥1 for q=0,1,2,3. Put

a=3(2t²+(1−2t)²),   b=2(w0²+w1²+w2²)/3−1.

**Theorem:** Every feasible scalar system has ab>1201/1250. The explicit rational R=2, λ1=λ2=1/2 system below has all four covering sums exactly one, b>0 and ab=1150153/1253320<1201/1250. The strict comparison covers **all** t∈[0,1/2] and **all real** boundary weights, not just a finite sample.

## Exact scalar optimization

Let u=1−2t, E=2t²+u², D=t²+u² and S=w0²+w1²+w2². The four scalar conditions are exactly

C0=t*w0+u*w1+t*w2≥1;
C1=t*w1+u*w2+t≥1;
C2=t*w2+u+t≥1;
C3=1.

For t>0, C2 is w2≥1. Since a=3E>0 is fixed, minimizing ab means minimizing the strictly convex S over these constraints.

**Case 0<t<1/3.** The unique optimum is

w=(t(1−t)/D, u(1−t)/D, 1).

Here C0=C2=C3=1, while C1=1+t²(1−3t)/D>1. The nonnegative q0 and q2 dual multipliers are α=2(1−t)/D and β=2u(1−3t)/(tD). For any feasible v,

||v||²−||w||² = ||v−w||² + α(C0(v)−1)+β*t(v2−1)≥||v−w||².

The exact optimum is ab=1−2t²(1−3t)²/D. Set h=t(1−3t)/u. Because D≥u² and 150t²−64t+7>0 (discriminant −104), h<7/50. Hence ab≥1−2h²>1201/1250.

**Case 1/3≤t≤1/2.** Define c0=(t,u,t), c1=(0,t,u), d=(1,1−t). The Gram matrix G=(ci·cj) is positive definite because c0,c1 are independent. Let Δ=det G=14t⁴−28t³+23t²−8t+1. Put

μ0=(D−2tu(1−t))/Δ>0,
μ1=−(3t−1)(2t²−4t+1)/Δ≥0,
w=μ0*c0+μ1*c1.

The inequalities are satisfied, with C0=C1=1; also w2−1=−t²(2t−1)(3t−1)/Δ≥0. Positivity of μ0 follows from D≥2tu and 1−t<1; μ1≥0 follows because 2t²−4t+1<0 on [1/3,1/2]. For any other feasible v, the exact KKT identity gives

||v||²−||w||²=||v−w||²+2μ0*c0·(v−w)+2μ1*c1·(v−w)≥||v−w||².

Thus this is the unique global minimum, with

ab=1−2t²(3t−1)²(10t²−12t+3)/Δ.

Write P(t)=10t²−12t+3. P is decreasing on [1/3,1/2] and P(9/25)=−3/125. Thus when t≥9/25, ab≥1. Otherwise t∈[1/3,9/25], and 0≤P≤1/9, t≤9/25, 3t−1≤2/25. By the Gram cross-product formula Δ≥t⁴≥1/81, whence

ab≥1−2*(9/25)²*(2/25)²*(1/9)*81=1−5832/390625>1201/1250.

At t=1/3 the optimum is w=(1,1,1), ab=1; at t=1/2 it is w=(1,1,1), ab=3/2. At t=0 the constraints reduce to w1,w2≥1, giving optimal w=(0,1,1), ab=1. All optimizers have b>0, so this argument never invokes the theorem in an ineligible b≤0 regime: in the upper branch, C0≥1 yields S≥1/E≥2, while the lower branch has b=(1−3t²)/(3D)>0.

## Fixed rational mixed certificate

Use kernels p1=(3/20,7/10,3/20), p2=(17/60,13/30,17/60), weights λ1=λ2=1/2, and boundary vectors

w1=(211401/783325,214053/156665,700569/783325),
w2=(399313/783325,153773/156665,827137/783325).

Direct substitution gives (C0,C1,C2,C3)=(1,1,1,1), a=53/40, b=21701/31333>0 and ab=1150153/1253320. Hence it **strictly beats every scalar m=3, L=1 system**, although its coefficient sqrt(ab)≈0.958 is weaker than Hou–Zhao's published 0.94349….

## What is and is not verified

A 20 November local SymPy worker established both polynomial identities and reconstructed all rational covers, with a corrupted-witness rejection. Independently, a new Fraction/Cramer's-rule checker evaluated **all 190** fixed equal-mixture grid pairs (1≤k≤l≤19, t=k/60), and matched the exact 19 November complete-row SHA256 digest 199215f53ae24a71df8820cc226d597d5a6b6c1a9c26b1083ac68cf7904726a1. The fresh checker orders Fractions numerically, avoiding the earlier invalid scalar string-ordering diagnostic. That grid certificate is finite; the real-parameter theorem follows from the ordinary proof here, not extrapolation. No global optimum for R≥2 or larger m is claimed. No Lean, peer-review or literature-priority verification is claimed.
