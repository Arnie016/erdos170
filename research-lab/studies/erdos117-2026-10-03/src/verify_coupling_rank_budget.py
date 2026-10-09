from pathlib import Path
PUBLICATION_ROOT = Path(__file__).resolve().parents[1]
(PUBLICATION_ROOT / 'results').mkdir(parents=True, exist_ok=True)
import itertools, json, time

def gf2_rank(rows, ncols):
    rows = [r for r in rows if r]
    rank = 0
    for col in range(ncols-1, -1, -1):
        pivot = next((i for i in range(rank, len(rows)) if (rows[i] >> col) & 1), None)
        if pivot is None: continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for i in range(len(rows)):
            if i != rank and ((rows[i] >> col) & 1): rows[i] ^= rows[rank]
        rank += 1
        if rank == len(rows): break
    return rank

def basis_of_vectors(vs, n):
    basis=[]
    for v in vs:
        if gf2_rank(basis+[v], n) > len(basis): basis.append(v)
    return basis

def span_set(basis):
    vals={0}
    for b in basis: vals |= {x ^ b for x in list(vals)}
    return frozenset(vals)

def all_subspaces(n):
    seen={frozenset({0}): []}
    nonzero=list(range(1,1<<n))
    for k in range(1,n+1):
        for comb in itertools.combinations(nonzero,k):
            b=basis_of_vectors(comb,n)
            if len(b)!=k: continue
            seen.setdefault(span_set(b),b)
    return list(seen.items())

def symplectic_rows(n):
    assert n%2==0
    rows=[0]*n
    for i in range(0,n,2):
        rows[i] |= 1<<(i+1); rows[i+1] |= 1<<i
    return rows

def alternating_rows(n, mask):
    rows=[0]*n; bit=0
    for i in range(n):
        for j in range(i+1,n):
            if (mask>>bit)&1:
                rows[i] |= 1<<j; rows[j] |= 1<<i
            bit += 1
    return rows

def form(A,x,y):
    z=0
    for i,row in enumerate(A):
        if (x>>i)&1: z ^= ((row & y).bit_count() & 1)
    return z

def restriction_rank(A,basis,nambient):
    k=len(basis); rows=[]
    for x in basis:
        row=0
        for j,y in enumerate(basis):
            if form(A,x,y): row |= 1<<j
        rows.append(row)
    return gf2_rank(rows,k)

def nullspace_basis(equations,n):
    vals=[x for x in range(1<<n) if all(((e & x).bit_count() & 1)==0 for e in equations)]
    return basis_of_vectors(vals,n)

def M_times_t(rowsM,t):
    v=0
    for i,row in enumerate(rowsM):
        if ((row & t).bit_count() & 1): v |= 1<<i
    return v

def check_case(r,b):
    ns,nt=2*r,2*b; sig=symplectic_rows(ns); tau=symplectic_rows(nt)
    subs=all_subspaces(nt); gamma_bits=nt*(nt-1)//2
    total=0; min_margin_S=999; min_margin_T=999; worst=None
    for gmask in range(1<<gamma_bits):
        gam=alternating_rows(nt,gmask); gr=gf2_rank(gam,nt)
        assert gr%2==0
        u=gr//2; want_dim=nt-u; candidates=[]
        for Sset,basis in subs:
            if len(basis)!=want_dim: continue
            if all(form(gam,x,y)==0 for x in basis for y in basis): candidates.append(basis)
        if not candidates: raise AssertionError(('no isotropic L',r,b,gmask,u))
        for L in candidates:
            rt=restriction_rank(tau,L,nt); marginT=rt-(2*b-2*u)
            min_margin_T=min(min_margin_T,marginT)
            if marginT<0: raise AssertionError(('target rank fail',r,b,gmask,u,L,rt))
        L=candidates[0]
        for mmask in range(1<<(ns*nt)):
            rowsM=[(mmask>>(i*nt)) & ((1<<nt)-1) for i in range(ns)]
            q=gf2_rank(rowsM,nt)
            equations=[M_times_t(rowsM,l) for l in L]
            K=nullspace_basis(equations,ns); rs=restriction_rank(sig,K,ns)
            marginS=rs-(2*r-2*q)
            if marginS<min_margin_S:
                min_margin_S=marginS; worst=(r,b,gmask,mmask,q,len(K),rs)
            if marginS<0: raise AssertionError(('source rank fail',r,b,gmask,mmask,q,K,rs))
            total += 1
    return {'r':r,'b':b,'cases':total,'min_source_margin':min_margin_S,'min_target_margin':min_margin_T,'worst_source':worst}

start=time.time(); results=[check_case(2,1),check_case(1,2)]
out={'status':'PASS','domain':'all alternating gamma and all binary cross matrices for (r,b)=(2,1),(1,2)','results':results,'total_cases':sum(x['cases'] for x in results),'wall_seconds':time.time()-start}
(PUBLICATION_ROOT/'coupling_rank_budget_regression.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
