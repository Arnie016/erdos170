#!/usr/bin/env python3
"""Independent exact regression for the 2026-10-20 first-denominator theorem.

Acceptance frozen before executing:
- t in {7^1,...,7^6, 7*13}, n=t*t, x0=(n+3)//4, x1=x0+1.
- Check each prime factor of t and x0 is 1 mod 3 (no reliance on factor theorem).
- Verify exact integer identity for a strict ordered solution with first denominator x1.
- Negative control n=25 must admit a strict solution with first denominator x0.
- No conclusion for untested t; the ordinary proof establishes the universal statement.
"""
import json, math, hashlib, sys, time

def distinct_prime_factors(v):
    n=v
    out=[]
    d=2
    while d*d<=n:
        if n%d==0:
            out.append(d)
            while n%d==0:n//=d
        d+=1 if d==2 else 2
    if n>1:out.append(n)
    return out

def equality_exact(n,x,y,z):
    return 4*x*y*z == n*(x*y+x*z+y*z)

def verify_family(t):
    assert t%2==1 and t%7==0
    ps_t=distinct_prime_factors(t)
    assert all(p%3==1 for p in ps_t)
    n=t*t
    assert n%24==1
    x0=(n+3)//4
    k=(t-1)//2
    assert x0 == k*k+k+1
    ps_x0=distinct_prime_factors(x0)
    assert all(p%3==1 for p in ps_x0)
    m=n*x0
    assert m%3==1
    x1=x0+1
    assert x1==(n+7)//4
    assert (n*x1)%7==0
    q=(n*x1)//7
    y,z=q+1,q*(q+1)
    assert x1<y<z
    assert equality_exact(n,x1,y,z)
    return {'t':t,'n':n,'x0_excluded':x0,'x1_attained':x1,'primes_t':ps_t,'primes_x0':ps_x0,'witness':{'x':x1,'y':y,'z':z},'integer_identity_passed':True}

def control():
    n=25;x0=(n+3)//4
    m=n*x0
    p=5
    assert m%3==1
    assert m%p==0
    y=(m+p)//3;z=(m+m*m//p)//3
    assert (m+p)%3==0 and (m+m*m//p)%3==0
    assert x0<y<z and equality_exact(n,x0,y,z)
    return {'n':n,'x0':x0,'witness':{'x':x0,'y':y,'z':z},'identity_passed':True}

if __name__=='__main__':
    st=time.monotonic()
    samples=[7**j for j in range(1,7)]+[7*13]
    rec={'domain':'exactly t=7^j for j=1..6, plus t=91; negative control n=25',
         'seed':None,'basis':'integer arithmetic; trial-division prime factorization',
         'cases':[verify_family(t) for t in samples], 'negative_control':control()}
    rec['wall_time_seconds']=round(time.monotonic()-st,5)
    print(json.dumps(rec,sort_keys=True,indent=2))
