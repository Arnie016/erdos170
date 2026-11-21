#!/usr/bin/env python3
"""Exact companion matrix resultant. Frozen before execution."""
import json,hashlib,time,sys
from pathlib import Path
N=517
cases={"target_P4":[1,0,0,1,1,1,0,0,1],"positive_P3":[1,0,1,1,1,0,1],"negative_Phi11":[1]*11}
def matmul(A,B):
 n=len(A); Z=[[0]*n for _ in range(n)]
 for i in range(n):
  for k in range(n):
   a=A[i][k]
   if a:
    for j in range(n): Z[i][j]+=a*B[k][j]
 return Z
def matexp(A,p):
 n=len(A); X=[[int(i==j) for j in range(n)] for i in range(n)]
 while p:
  if p&1:X=matmul(X,A)
  p//=2
  if p:A=matmul(A,A)
 return X
def bareiss(A):
 A=[row[:] for row in A];n=len(A); sign=1;prev=1
 for k in range(n-1):
  pivot=k
  while pivot<n and A[pivot][k]==0:pivot+=1
  if pivot==n:return 0
  if pivot!=k:A[k],A[pivot]=A[pivot],A[k];sign=-sign
  val=A[k][k]
  for i in range(k+1,n):
   for j in range(k+1,n):
    num=A[i][j]*val-A[i][k]*A[k][j]
    assert num%prev==0
    A[i][j]=num//prev
  for i in range(k+1,n):A[i][k]=0
  prev=val
 return sign*A[n-1][n-1]
def res(f):
 d=len(f)-1;assert f[-1]==1 and d%2==0
 C=[[0]*d for _ in range(d)]
 for j in range(d-1):C[j+1][j]=1
 for i in range(d):C[i][d-1]=-f[i]
 M=matexp(C,N)
 for i in range(d):M[i][i]-=1
 det=bareiss(M);val=sum(f);assert det%val==0
 return dict(determinant=str(det), resultant=str(det//val),f_at_one=val)
t0=time.monotonic();out={k:res(f) for k,f in cases.items()}
assert out['positive_P3']['resultant']=='1';assert out['negative_Phi11']['resultant']=='0'
result={'method':'companion matrix and Bareiss','N':N,'cases':out,'internal_seconds':time.monotonic()-t0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
