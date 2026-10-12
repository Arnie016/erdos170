#!/usr/bin/env python3
"""Independent rank-incidence checker for SHA256 fixed samples, matrix rows not wedge evaluation."""
import hashlib,json,platform,time
start=time.monotonic()
PAIRS=[(r,c) for r in range(5) for c in range(r+1,5)]

def rank(vs):
  b={}
  for x in vs:
    while x:
      i=x.bit_length()-1
      if i in b:x^=b[i]
      else:b[i]=x;break
  return len(b)

def matrix(bits):
  mat=[0]*5
  for k,(a,b) in enumerate(PAIRS):
    if bits>>k&1:mat[a]^=1<<b;mat[b]^=1<<a
  return mat

def left_action(v,M):
  z=0
  for k in range(5):
    if v>>k&1:z^=M[k]
  return z

hist={};checked=0;first=None;errors=[]
for idx in range(1024):
  h=hashlib.sha256(f'erdos117-20261012-{idx}'.encode()).digest()
  triples=[int.from_bytes(h[2*j:2*j+2],'little') & 1023 for j in range(3)]
  if rank(triples)!=3:continue
  matrices=[matrix(x) for x in triples]
  ctr={0:0,1:0,2:0,3:0}
  for v in range(1,32):
    rows=[left_action(v,M) for M in matrices]
    ctr[rank(rows)]+=1
  if ctr[0]:continue
  t2=0
  for lam in range(1,8):
    M=[0]*5
    for j in range(3):
      if lam>>j&1:M=[a^b for a,b in zip(M,matrices[j])]
    rr=rank(M)
    assert rr in (2,4), (idx,rr)
    t2+=(rr==2)
  checked+=1
  left=3*ctr[1]+ctr[2]
  right=7+6*t2
  if left!=right or ctr[2]%3!=1:errors.append({'index':idx,'ctr':ctr,'t2':t2,'left':left,'right':right})
  key=(t2,ctr[1],ctr[2])
  hist[str(key)]=hist.get(str(key),0)+1
  if first is None or (t2==0 and first['t2']!=0):first={'index':idx,'forms':triples,'ctr':ctr,'t2':t2,'lhs':left,'rhs':right}
  if idx==0:
    assert all(M[a]>>a&1==0 for M in matrices for a in range(5))
    assert all((M[a]>>b&1)==(M[b]>>a&1) for M in matrices for a in range(5) for b in range(5))
assert checked==1007,checked
assert not errors,errors
assert first['lhs']+1 != first['rhs']
print(json.dumps({'result':'PASS','domain':'full-rank faithful triple from 1024 deterministic indices','checked':checked,'hist':hist,'representative':first,'negative_control':'n2 increment rejected','errors':errors,'wall_seconds':time.monotonic()-start,'python':platform.python_version()}))
