#!/usr/bin/env python3
"""Finite controls; no claim of a universal computational proof."""
import itertools,json,time
def build(xs,mul):
 d={x:i for i,x in enumerate(xs)};t=[[d[mul(a,b)] for b in xs] for a in xs];n=len(t)
 e=next(i for i in range(n) if all(t[i][j]==j==t[j][i] for j in range(n)))
 assert all(t[t[i][j]][k]==t[i][t[j][k]] for i in range(n) for j in range(n) for k in range(n))
 assert all(any(t[i][j]==e==t[j][i] for j in range(n)) for i in range(n))
 return t,e
def d8():return build([(i,j) for i in range(4) for j in range(2)],lambda a,b:((a[0]+(-1 if a[1] else 1)*b[0])%4,(a[1]+b[1])%2))
def q8():
 b={(1,2):(1,3),(2,3):(1,1),(3,1):(1,2),(2,1):(-1,3),(3,2):(-1,1),(1,3):(-1,2)}
 def mul(a,c):
  s=a[0]*c[0];i=a[1];j=c[1]
  if i==0:return s,j
  if j==0:return s,i
  if i==j:return -s,0
  z,k=b[i,j];return s*z,k
 return build([(s,i) for s in (-1,1) for i in range(4)],mul)
def s3():return build(list(itertools.permutations(range(3))),lambda a,b:tuple(a[b[i]] for i in range(3)))
def c4():return build(list(range(4)),lambda a,b:(a+b)%4)
def calc(t,e):
 n=len(t); comm=lambda x,y:t[x][y]==t[y][x]
 Z=[i for i in range(n) if all(comm(i,j) for j in range(n))]
 omega=max(sum(mask>>i&1 for i in range(n)) for mask in range(1,1<<n) if all(not comm(i,j) for i in range(n) for j in range(i+1,n) if (mask>>i&1) and (mask>>j&1)))
 inv=[next(j for j in range(n) if t[i][j]==e==t[j][i]) for i in range(n)]
 groups=[]
 for mask in range(1,1<<n):
  if not(mask>>e&1):continue
  s=[i for i in range(n) if mask>>i&1]
  if all(mask>>inv[i]&1 for i in s) and all(mask>>t[i][j]&1 and comm(i,j) for i in s for j in s):groups.append(mask)
 cover={0:0}
 for m in range(1<<n):
  if m not in cover:continue
  for g in groups:
   k=m|g;cover[k]=min(cover.get(k,n+1),cover[m]+1)
 return omega,cover[(1<<n)-1],n//len(Z)
if __name__=="__main__":
 start=time.monotonic();want={'D8':(3,3,4),'Q8':(3,3,4),'S3':(4,4,6),'C4':(1,1,1)};out={}
 for name,fn in [('D8',d8),('Q8',q8),('S3',s3),('C4',c4)]:
  t,e=fn();got=calc(t,e);assert got==want[name],(name,got);out[name]=got
 print(json.dumps({'status':'PASS','cases':out,'seconds':time.monotonic()-start}))
