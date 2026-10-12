#!/usr/bin/env python3
"""Bounded pre-registered deterministic 1024 sample probes of rank-3 binary alternating forms.
Negative conditions for target: no isotropic 3-space and no 11-clique.
Finite order: i=0..1023, SHA256('erdos117-20261012-'+str(i)) produces three 10-bit forms.
"""
import hashlib,sys,time,json,platform
start=time.monotonic()
PAIRS=[(i,j) for i in range(5) for j in range(i+1,5)]

def wedge(a,b):
  return sum((((a>>i)&1)&((b>>j)&1) ^ (((a>>j)&1)&((b>>i)&1)))<<k for k,(i,j) in enumerate(PAIRS)
W=[[wedge(a,b) for b in range(32)] for a in range(32)]

def rank(x):
  basis=[]
  for z in x:
    for b in basis: z=min(z,z^b)
    if z:
      basis.append(z);basis.sort(reverse=True)
  return len(basis)

def sample(i):
  s=hashlib.sha256(('erdos117-20261012-'+str(i)).encode()).digest()
  return [int.from_bytes(s[j*2:j*2+2],'little')&1023 for j in range(3)]

def make_b(tab):
  vals=[0]*1024
  for k in range(1024):
    vals[k]=sum(((k&r).bit_count()%2)<<i for i,r in enumerate(tab))
  return [[vals[W[a][b]] for b in range(32)] for a in range(32)]

def maxclique_or_at_least_11(b):
  adj=[0]*32
  for v in range(1,32):
    adj[v]=sum(1<<(w-1) for w in range(1,32) if w!=v and b[v][w])
  nodes=0
  def coloring(P):
    order=[];bounds=[];color=0
    while P:
      color+=1; available=P
      while available:
        bit=available&-available; v=bit.bit_length(); order.append(v);bounds.append(color)
        P ^= bit; available^=bit;available &= ~adj[v]
    return order,bounds
  best=0
  def dfs(P,sz):
    nonlocal best,nodes
    nodes+=1
    if not P:
      if sz>best:best=sz
      return best>=11
    order,bounds=coloring(P)
    for idx in range(len(order)-1,-1,-1):
      if sz+bounds[idx]<=max(best,10):return False
      v=order[idx]
      if dfs(P & adj[v],sz+1):return True
      P &= ~(1<<(v-1))
    return False
  ge11=dfs((1<<31)-1,0)
  return ge11,best,nodes

def iso3(b):
  for a in range(1,32):
    for c in range(a+1,32):
      if b[a][c]!=0:continue
      ac=a^c
      for d in range(1,32):
        if d in (a,c,ac):continue
        if b[a][d]==0 and b[c][d]==0:return (a,c,d)
  return None

hist={}; complete=0; examined=0; eligible=0; events=[]
for i in range(1024):
  x=sample(i)
  if rank(x)<3:continue
  b=make_b(x)
  if i==0:
    assert all(b[j][j]==0 and b[j][k]==b[k][j] for j in range(32) for k in range(32))
    assert all(b[j^k][m]==(b[j][m]^b[k][m]) for j in range(32) for k in range(32) for m in range(32))
  examined+=1
  radical=[v for v in range(1,32) if all(b[v][w]==0 for w in range(32))]
  if radical:hist['radical'] = hist.get('radical',0)+1;continue
  eligible+=1
  iso=iso3(b)
  if iso:hist['iso3']=hist.get('iso3',0)+1;continue
  g11,best,nodes=maxclique_or_at_least_11(b)
  complete+=1
  hist['clique_ge_11' if g11 else 'no_clique_11']=hist.get('clique_ge_11' if g11 else 'no_clique_11',0)+1
  if not g11:
    events.append({'i':i,'forms':x,'omega_at_most':10,'iso3':None,'nodes':nodes})
    print(json.dumps({'status':'POTENTIAL_CERTIFICATE','candidate':events[-1],'hist':hist,'elapsed':time.monotonic()-start}),flush=True)
    break
  if time.monotonic()-start>30:break
print(json.dumps({'status':'COMPLETE' if examined==1024 else 'EARLY_STOP','indices_processed':i+1,'full_rank':examined,'faithful':eligible,'no_iso3_cases':complete,'hist':hist,'candidate':events,'wall_seconds':time.monotonic()-start,'python':platform.python_version()}))
