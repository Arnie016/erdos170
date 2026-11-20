#!/usr/bin/env python3
"""Independent 190-case audit via determinant-based rational QP active sets."""
from fractions import Fraction as Q
from itertools import combinations
import hashlib, json

def det(M):
 n=len(M)
 if n==0:return Q(1)
 if n==1:return M[0][0]
 return sum(((-1)**j)*M[0][j]*det([r[:j]+r[j+1:] for r in M[1:]]) for j in range(n))

def cram(M,d):
 den=det(M)
 if den==0:return None
 n=len(d)
 return [det([[d[i] if j==k else M[i][j] for j in range(n)] for i in range(n)])/den for k in range(n)]

def dot(x,y):return sum((a*b for a,b in zip(x,y)),Q(0))

def calc(k,l):
 t0=Q(k,60);t1=Q(l,60)
 P=((t0,1-2*t0,t0),(t1,1-2*t1,t1))
 rows=[];need=[]
 for q in (0,1,2):
  coeff=tuple(P[r][j-q]/2 if 0<=j-q<3 else Q(0) for r in (0,1) for j in (0,1,2))
  fixed=sum((P[r][i]/2 for r in (0,1) for i in range(3) if q+i>=3),Q(0))
  rows.append(coeff);need.append(1-fixed)
 solutions=[]
 for size in range(4):
  for subset in combinations(range(3),size):
   if not subset:
    v=(Q(0),)*6
   else:
    M=[[dot(rows[i],rows[j]) for j in subset] for i in subset]
    y=cram(M,[need[i] for i in subset])
    if y is None or min(y)<0:continue
    v=tuple(sum((y[a]*rows[subset[a]][j] for a in range(size)),Q(0)) for j in range(6))
   if any(dot(rows[i],v)<need[i] for i in range(3)):continue
   energy=3*sum((x*x for pp in P for x in pp),Q(0))/2
   boundary=sum((x*x for x in v),Q(0))/3-1
   assert boundary>0
   ab=energy*boundary
   covers=tuple(dot(rows[i],v)+1-need[i] for i in range(3))+(Q(1),)
   solutions.append((dot(v,v),ab,v,tuple(subset),covers))
 assert solutions
 s,ab,v,active,covers=min(solutions,key=lambda z:z[0])
 return dict(ab=ab,v=v,active=active,covers=covers,p=P)

rows=[];winner=None
for k in range(1,20):
 for l in range(k,20):
  x=calc(k,l)
  rows.append([k,l,str(x['ab'])])
  if winner is None or x['ab']<winner[0]:winner=(x['ab'],k,l,x)
assert len(rows)==190
ab,k,l,x=winner
assert (k,l)==(9,17) and ab==Q(1150153,1253320)
assert x['covers']==(1,1,1,1)
assert calc(12,12)['ab']==Q(121,125)
best_scalar=min((calc(k,k)['ab'],k) for k in range(1,20))
assert best_scalar[0]>ab
bad=list(x['v']);bad[0]-=Q(1,100)
full=[]
for q in range(4):
 full.append(sum((x['p'][r][i]*(bad[3*r+q+i] if q+i<3 else 1)/2 for r in (0,1) for i in range(3)),Q(0)))
assert full[0]<1 and tuple(full[1:])==(1,1,1)
wbad=[Q(0)]*6
badcover=[sum((x['p'][r][i]*(wbad[3*r+q+i] if q+i<3 else 1)/2 for r in (0,1) for i in range(3)),Q(0)) for q in range(4)]
assert badcover[0]<1 and badcover[2]<1 and badcover[3]==1
canonical=json.dumps(rows,separators=(',',':')).encode()
print(json.dumps({'claim':'SIDON-3BIN-FULLSCALAR-2026-11-20','scope':'190 fixed pair QPs, k<=l in 1..19','cases':len(rows),'grid_sha256':hashlib.sha256(canonical).hexdigest(),'best':{'k':k,'l':l,'ab':str(ab),'covers':[str(v) for v in x['covers']],'weights':[str(v) for v in x['v']],'active':x['active']},'best_scalar_grid':{'k':best_scalar[1],'ab':str(best_scalar[0])},'negative_controls':'reject_q0_and_q2','grid_complete':True},sort_keys=True))
