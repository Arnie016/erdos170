#!/usr/bin/env python3
"""Independent modular-polynomial exponentiation and Sylvester determinant."""
import json,hashlib,time,sys
from pathlib import Path
N=517
cases={"target_P4":[1,0,0,1,1,1,0,0,1],"positive_P3":[1,0,1,1,1,0,1],"negative_Phi11":[1]*11}
def clean(a):
 while len(a)>1 and a[-1]==0:a.pop()
 return a
def pmul(a,b,f):
 d=len(f)-1;t=[0]*(len(a)+len(b)-1)
 for i,x in enumerate(a):
  for j,y in enumerate(b):t[i+j]+=x*y
 for k in range(len(t)-1,d-1,-1):
  x=t[k]
  if x:
   for j in range(d):t[k-d+j]-=x*f[j]
 return clean(t[:d] or [0])
def pexp_x(n,f):
 a=[1];b=[0,1]
 while n:
  if n%2:a=pmul(a,b,f)
  n//=2
  if n:b=pmul(b,b,f)
 return a
def bareiss(mat):
 M=[list(r) for r in mat];n=len(M);previous=1;sig=1
 for k in range(n-1):
  p=next((i for i in range(k,n) if M[i][k]),None)
  if p is None:return 0
  if p!=k:M[p],M[k]=M[k],M[p];sig=-sig
  v=M[k][k]
  for i in range(k+1,n):
   for j in range(k+1,n):
    w=M[i][j]*v-M[i][k]*M[k][j]
    assert w%previous==0
    M[i][j]=w//previous
  for i in range(k+1,n):M[i][k]=0
  previous=v
 return sig*M[-1][-1]
def resultant(f,r):
 f=clean(f[:]);r=clean(r[:]);a=len(f)-1;b=len(r)-1
 if r==[0]:return 0
 n=a+b;fh=list(reversed(f));rh=list(reversed(r));M=[]
 for i in range(b):M.append([0]*i+fh+[0]*(n-i-a-1))
 for i in range(a):M.append([0]*i+rh+[0]*(n-i-b-1))
 assert len(M)==n and all(len(row)==n for row in M)
 return bareiss(M)
def compute(f):
 r=pexp_x(N,f);r[0]-=1;r=clean(r)
 det=resultant(f,r);val=sum(f);assert det%val==0
 return dict(determinant=str(det),resultant=str(det//val),f_at_one=val,remainder_x517_minus1=r)
t0=time.monotonic();out={k:compute(f) for k,f in cases.items()}
assert out['positive_P3']['resultant']=='1';assert out['negative_Phi11']['resultant']=='0'
result={'method':'modular polynomial powers and Sylvester Bareiss','N':N,'cases':out,'internal_seconds':time.monotonic()-t0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:out[k]['resultant'] for k in out}))
