#!/usr/bin/env python3
"""Reproducible independent arithmetic regression; finite checks are not the proof."""
import json, math, time
from fractions import Fraction

def phi_sieve(N):
    phi=list(range(N+1))
    for p in range(2,N+1):
        if phi[p]==p:
            for k in range(p,N+1,p):
                phi[k]-=phi[k]//p
    return phi

def primes_upto(N):
    f=bytearray(b'\x01')*(N+1)
    f[:2]=b'\x00\x00'
    for p in range(2,math.isqrt(N)+1):
        if f[p]: f[p*p:N+1:p]=b'\x00'*(((N-p*p)//p)+1)
    return [x for x in range(2,N+1) if f[x]]

def divisors(N):
    return [d for d in range(1,N+1) if N%d==0]

def local_h(p, in_u, in_a):
    if p==2:
        if not in_u and not in_a: return Fraction(1,2)
        if in_u and in_a: return Fraction(-1,8)
        return Fraction(0,1)
    if not in_u and not in_a: return Fraction(1,1)
    if in_u and in_a: return Fraction(-(p+1),p*p*(p-1))
    return Fraction(1,p*(p-1))

def h(d,e,primes):
    result=Fraction(1,1)
    for p in primes:
        if d%(p*p)==0 or e%(p*p)==0: return Fraction(0,1)
        result *= local_h(p, d%p==0, e%p==0)
    return result

start=time.perf_counter()
N=3000
phi=phi_sieve(max(N,4*32*32))
for u in range(1,33):
    for a in range(1,33):
        actual=Fraction(1,phi[4*u*a]) if math.gcd(u,a)==1 else Fraction(0,1)
        fact=Fraction(1,(2 if u*a%2 else 4)*phi[u]*phi[a]) if math.gcd(u,a)==1 else Fraction(0,1)
        assert actual==fact,(u,a,actual,fact)
primes=primes_upto(16)
for u in range(1,17):
    for a in range(1,17):
        reconstructed=sum((h(d,e,primes)*Fraction(d*e,u*a) for d in divisors(u) for e in divisors(a)),Fraction(0,1))
        direct=Fraction(1,phi[4*u*a]) if math.gcd(u,a)==1 else Fraction(0,1)
        assert reconstructed==direct,(u,a,reconstructed,direct)
wrong=sum((Fraction(1,phi[4*u*a]) for u in range(1,9) for a in range(1,9)),Fraction(0,1))
right=sum((Fraction(1,phi[4*u*a]) for u in range(1,9) for a in range(1,9) if math.gcd(u,a)==1),Fraction(0,1))
assert wrong!=right
assert Fraction(1,2)-Fraction(1,8)==Fraction(3,8)
inv=[0.0]+[1.0/v for v in phi[1:N+1]]
selected={100,300,1000,3000}; values={}
for u in range(1,N+1):
    for a in range(1,N+1):
        if math.gcd(u,a)!=1: continue
        contribution=inv[u]*inv[a]*(0.5 if (u*a)%2 else 0.25)
        if u<=100 and a<=100: values[100]=values.get(100,0)+contribution
        if u<=300 and a<=300: values[300]=values.get(300,0)+contribution
        if u<=1000 and a<=1000: values[1000]=values.get(1000,0)+contribution
        values[3000]=values.get(3000,0)+contribution
constant=9/(2*math.pi**2)
reported={100:0.62795,300:0.59503,1000:0.57114,3000:0.55559}
ratios={str(J):{'sum':values[J],'ratio':values[J]/math.log(J)**2,'paper_ratio':reported[J],'delta':values[J]/math.log(J)**2-reported[J]} for J in sorted(selected)}
for J in selected: assert abs(ratios[str(J)]['delta'])<.001,(J,ratios[str(J)])
prime_product=(3/8)*math.prod(1+1/(q*q) for q in primes_upto(10000) if q!=2)
assert abs(prime_product-constant)<1e-5
print(json.dumps({'date':'2026-10-22','status':'PASS','domain':{'parity':'all 1<=u,a<=32','convolution':'all 1<=u,a<=16','benchmark':[100,300,1000,3000],'prime_product':'odd primes <=10000'},'leading_constant_formula':'9/(2*pi*pi)','leading_constant':constant,'prime_product_10000':prime_product,'checks':{'parity_totient':1024,'convolution':256,'negative_control':True},'values':ratios,'wall_seconds':time.perf_counter()-start},indent=2))
