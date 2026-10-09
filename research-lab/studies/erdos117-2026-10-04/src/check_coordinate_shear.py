import json, time
from pathlib import Path
PUBLICATION_ROOT=Path(__file__).resolve().parents[1]

def gf2_rank(rows,ncols):
    rows=[r & ((1<<ncols)-1) for r in rows if r]; rank=0; col=ncols-1
    while col>=0 and rank<len(rows):
        pivot=next((i for i in range(rank,len(rows)) if (rows[i]>>col)&1),None)
        if pivot is None: col-=1; continue
        rows[rank],rows[pivot]=rows[pivot],rows[rank]
        for i in range(len(rows)):
            if i!=rank and ((rows[i]>>col)&1): rows[i]^=rows[rank]
        rank+=1; col-=1
    return rank

def mat_rows(n,pairs):
    rows=[0]*n
    for i,j in pairs: rows[i]^=1<<j; rows[j]^=1<<i
    return rows

def restrict_rows(M,basis,n):
    k=len(basis); out=[0]*k
    for i,vi in enumerate(basis):
        Mvi=0
        for r in range(n):
            if (vi>>r)&1: Mvi^=M[r]
        for j,vj in enumerate(basis):
            if (Mvi&vj).bit_count()&1: out[i]|=1<<j
    return out

def is_isotropic(M,basis,n): return gf2_rank(restrict_rows(M,basis,n),len(basis))==0

start=time.time(); results=[]
for a in range(1,9):
 for b in range(1,a+1):
    n=2*a+b+2; X=list(range(a));Y=list(range(a,2*a));Z=list(range(2*a,2*a+b));U=2*a+b;W=U+1
    B0=mat_rows(n,[(X[i],Y[i]) for i in range(a)])
    B1=mat_rows(n,[(U,W)])
    B2=mat_rows(n,[(Y[i],Z[i]) for i in range(b)])
    B0s=[B0[i]^B2[i] for i in range(n)]
    rad=[(1<<X[i])^(1<<Z[i]) for i in range(b)]+[1<<U,1<<W]
    A0=[1<<j for j in Y]+rad
    A1=[1<<j for j in Y]+[(1<<X[i])^(1<<Z[i]) for i in range(b)]+[1<<U]
    psi_rows=[]
    for v in A1:
        mask=0
        for j,x in enumerate(X):
            if (B0s[x]&v).bit_count()&1:mask|=1<<j
        psi_rows.append(mask)
    rec={'a':a,'b':b,'ncols':n,'rank_beta0_old':gf2_rank(B0,n),'rank_beta0_sheared':gf2_rank(B0s,n),'radical_dim_expected':b+2,'radical_generators_rank':gf2_rank(rad,n),'radical_isotropic':is_isotropic(B0s,rad,n),'A0_dim':gf2_rank(A0,n),'A0_isotropic_beta0_sheared':is_isotropic(B0s,A0,n),'A0_quotient_dim_over_rad':gf2_rank(A0,n)-gf2_rank(rad,n),'rank_beta1_on_A0':gf2_rank(restrict_rows(B1,A0,n),len(A0)),'A1_dim':gf2_rank(A1,n),'A1_isotropic_beta0_sheared':is_isotropic(B0s,A1,n),'A1_isotropic_beta1':is_isotropic(B1,A1,n),'rank_beta2_on_A1':gf2_rank(restrict_rows(B2,A1,n),len(A1)),'interaction_rank_against_X':gf2_rank(psi_rows,a),'rank_beta0_sheared_on_old_T':gf2_rank(restrict_rows(B0s,[1<<j for j in Y[:b]+Z],n),2*b)}
    ok=(rec['rank_beta0_old']==2*a and rec['rank_beta0_sheared']==2*a and rec['radical_generators_rank']==b+2 and rec['A0_isotropic_beta0_sheared'] and rec['A0_quotient_dim_over_rad']==a and rec['rank_beta1_on_A0']==2 and rec['A1_isotropic_beta0_sheared'] and rec['A1_isotropic_beta1'] and rec['rank_beta2_on_A1']==2*b and rec['interaction_rank_against_X']==a and rec['rank_beta0_sheared_on_old_T']==2*b)
    rec['pass']=ok; results.append(rec)
out={'status':'PASS' if all(r['pass'] for r in results) else 'FAIL','domain':'all integer pairs 1<=b<=a<=8','cases':len(results),'wall_seconds':time.time()-start,'results':results}
(PUBLICATION_ROOT/'coordinate_shear_regression.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'status':out['status'],'cases':out['cases'],'wall_seconds':out['wall_seconds']}))
