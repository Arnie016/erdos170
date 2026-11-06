#!/usr/bin/env python3
"""Reproduce the explicit odd-q construction, prime sample only; NOT an all-q proof."""
import itertools,json
def pts(q):
 s=set()
 for p in itertools.product(range(q),repeat=3):
  if not any(p):continue
  a=next(x for x in p if x);inv=pow(a,-1,q)
  s.add(tuple(x*inv%q for x in p))
 return sorted(s)
def inner(x,y,q):return (x[0]*y[2]+x[2]*y[0]-2*x[1]*y[1])%q
def four(adj):
 n=len(adj);z=0
 for i in range(n):
  for j in range(i+1,n):
   k=len(adj[i]&adj[j]);z+=k*(k-1)//2
 assert z%2==0
 return z//2
results=[]
for q in (3,5,7,11,13,17):
 v=pts(q);m={p:i for i,p in enumerate(v)};g=[set() for _ in v]
 for i in range(len(v)):
  for j in range(i+1,len(v)):
   if inner(v[i],v[j],q)==0:g[i].add(j);g[j].add(i)
 assert len(v)==q*q+q+1 and sum(len(x) for x in g)//2==q*(q+1)**2//2
 assert four(g)==0
 ci, c0, cp, cm=(0,0,1),(1,0,0),(1,1,1),(1,q-1,1)
 a,b=(0,1,0),(1,0,q-1)
 assert len({ci,c0,cp,cm,a,b})==6
 assert inner(a,b,q)==0
 assert all(inner(x,y,q)!=0 for x,y in itertools.combinations((ci,c0,cp,cm),2))
 assert all(inner(x,y,q)==0 for x,y in ((ci,a),(c0,a),(a,b),(b,cp),(b,cm)))
 def add(x,y):i,j=m[x],m[y];g[i].add(j);g[j].add(i)
 def rm(x,y):i,j=m[x],m[y];g[i].remove(j);g[j].remove(i)
 add(ci,cp);assert four(g)==q-1
 add(c0,cm);assert four(g)==2*(q-1)
 rm(a,b);z=four(g);assert z==2*q-4
 assert sum(len(x) for x in g)//2==q*(q+1)**2//2+1
 results.append({"q":q,"n":len(v),"C4":z})
print(json.dumps(results))
