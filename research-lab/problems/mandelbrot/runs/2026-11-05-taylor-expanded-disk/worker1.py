#!/usr/bin/env python3
"""Frozen worker 1: exact Taylor coefficients in delta and outward modulus upper bounds.
Domain: c0=-3/4+i/2, j=0..6, rho in [1/1000,1/750,1/500,1/400,1/250,1/100].
No random sampling or external libraries.
"""
from fractions import Fraction as F
from math import isqrt,lcm
import json
from pathlib import Path
P=Path(__file__).parent
zero=(F(0),F(0)); one=(F(1),F(0)); c0=(F(-3,4),F(1,2))
rad=[F(1,1000),F(1,750),F(1,500),F(1,400),F(1,250),F(1,100)]
def plus(a,b): return a[0]+b[0],a[1]+b[1]
def mult(a,b): return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def sqnorm(a): return a[0]*a[0]+a[1]*a[1]
def ub(a):
    """Guaranteed strict rational upper for magnitude via upward integer sqrt at 10^-12 resolution."""
    d=lcm(a[0].denominator,a[1].denominator)
    x=int(a[0]*d); y=int(a[1]*d)
    return F(isqrt((x*x+y*y)*10**24)+1,d*10**12)
poly=[zero]
for step in range(1,7):
    v=[zero]*(2*len(poly)-1)
    for i,a in enumerate(poly):
        for j,b in enumerate(poly): v[i+j]=plus(v[i+j],mult(a,b))
    v[0]=plus(v[0],c0)
    if len(v)==1:v.append(one)
    else:v[1]=plus(v[1],one)
    poly=v
assert len(poly)==33 and poly[0]!=zero
assert poly[1]!=zero
w6=poly[0]
E=F(0); w=zero
for j in range(6):
    U=abs(w[0])+abs(w[1]); E=(2*U+E)*E+rad[0]
    w=plus(mult(w,w),c0)
assert w==w6
rows=[]
for rho in rad:
    power=rho; S=F(0)
    for coeff in poly[1:]:
        S+=ub(coeff)*power; power*=rho
    ok=sqnorm(w6)>(F(2)+S)**2
    rows.append({'rho':str(rho),'taylor_upper':str(S),'certifies':ok,'decimals':{'S':float(S),'margin':float(sqnorm(w6)-(F(2)+S)**2)}})
assert rows[0]['certifies'] and F(rows[0]['taylor_upper'])<E, 'Frozen baseline failed'
assert sqnorm(zero)<F(4), 'negative control c0=0 incorrectly certifies escape'
print(json.dumps({'method':'delta-Taylor recursion with rational outward-square-root bounds','degree':32,'w6':[str(x) for x in w6],'norm2':str(sqnorm(w6)),'old_E6':str(E),'rows':rows,'coefficients':[[str(x) for x in a] for a in poly],'negative_control':'c0=0 at delta=0 cannot pass escape test'},sort_keys=True))

