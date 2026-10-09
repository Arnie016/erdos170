import json,time
from pathlib import Path
PUBLICATION_ROOT=Path(__file__).resolve().parents[1]

def rank_gf2(rows,ncols):
    rows=[int(r) for r in rows if r];rank=0;col=0
    while col<ncols and rank<len(rows):
        piv=None;bit=1<<col
        for i in range(rank,len(rows)):
            if rows[i]&bit:piv=i;break
        if piv is None:col+=1;continue
        rows[rank],rows[piv]=rows[piv],rows[rank];pr=rows[rank]
        for i in range(len(rows)):
            if i!=rank and rows[i]&bit:rows[i]^=pr
        rank+=1;col+=1
    return rank

def matrix_rows_from_edges(n,edges):
    rows=[0]*n
    for i,j in edges:rows[i]^=1<<j;rows[j]^=1<<i
    return rows

def combine(B,mask):
    n=len(B[0]);out=[0]*n
    for k in range(3):
        if (mask>>k)&1:out=[out[i]^B[k][i] for i in range(n)]
    return out

def nullspace_basis(rows,n):
    A=[int(r) for r in rows];piv=[];r=0
    for c in range(n):
        bit=1<<c;p=next((i for i in range(r,n) if A[i]&bit),None)
        if p is None:continue
        A[r],A[p]=A[p],A[r]
        for i in range(n):
            if i!=r and A[i]&bit:A[i]^=A[r]
        piv.append(c);r+=1
        if r==n:break
    free=[c for c in range(n) if c not in piv];basis=[]
    for f in free:
        x=1<<f
        for i,p in enumerate(piv):
            if (A[i]>>f)&1:x|=1<<p
        basis.append(x)
    return basis

def bilinear(rowmat,x,y):
    s=0
    while x:
        lsb=x&-x;i=lsb.bit_length()-1
        s^=(rowmat[i]&y).bit_count()&1;x^=lsb
    return s

def restricted_rank(M,basis,n):
    rows=[]
    for x in basis:
        row=0
        for j,y in enumerate(basis):
            if bilinear(M,x,y):row|=1<<j
        rows.append(row)
    return rank_gf2(rows,len(basis))

def cross_rank(Mlam,Mmu,n):return restricted_rank(Mlam,nullspace_basis(Mmu,n),n)

def build(a,b):
    n=2*a+b+2;u=2*a+b;w=u+1
    edges=[[(i,a+i) for i in range(a)],[(u,w)],[(a+i,2*a+i) for i in range(b)]]
    return n,[matrix_rows_from_edges(n,e) for e in edges]

def expected_cross(lam,mu,a,b):
    c0=lam&1;c1=(lam>>1)&1;c2=(lam>>2)&1
    if mu==1:return 2*c1
    if mu==2:return 2*a if c0 else 2*b if c2 else 0
    if mu==3:return 0
    if mu==4:return 2*(a-b)*c0+2*c1
    if mu==5:return 2*c1
    if mu==6:return 2*(a-b)*c0
    if mu==7:return 0
    raise ValueError(mu)

start=time.time();cases=0;pair_checks=0;min_margin=None;failures=[]
for a in range(1,17):
 for b in range(1,a+1):
    n,B=build(a,b);forms={m:combine(B,m) for m in range(1,8)};xi=0
    for lam in range(1,8):
     for mu in range(lam+1,8):
        x=cross_rank(forms[lam],forms[mu],n);y=cross_rank(forms[mu],forms[lam],n)
        ex=expected_cross(lam,mu,a,b);ey=expected_cross(mu,lam,a,b)
        if (x,y)!=(ex,ey):failures.append({'a':a,'b':b,'lam':lam,'mu':mu,'actual':[x,y],'expected':[ex,ey]})
        xi=max(xi,(x+1)*(y+1));pair_checks+=1
    expected_xi=3*(2*a+1);margin=expected_xi-xi
    min_margin=margin if min_margin is None else min(min_margin,margin)
    if xi!=expected_xi:failures.append({'a':a,'b':b,'xi':xi,'expected_xi':expected_xi})
    cases+=1
out={'status':'PASS' if not failures else 'FAIL','domain':'all integer pairs 1<=b<=a<=16; all 21 unordered pairs of distinct nonzero scalar forms in F_2^3','parameter_cases':cases,'pair_checks':pair_checks,'formula_verified':'Xi(B)=3(2a+1)','closed_form_crossrank_table_verified':not failures,'minimum_xi_margin':min_margin,'wall_seconds':time.time()-start,'failures':failures[:20]}
(PUBLICATION_ROOT/'pencil_crossrank_regression.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
