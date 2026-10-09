import json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'results';OUT.mkdir(parents=True,exist_ok=True)
def bits4(v):return tuple((v>>i)&1 for i in range(4))
def f(case,v,w):
    x,y,z,t=bits4(v);xp,yp,zp,tp=bits4(w)
    return (x&yp,z&tp) if case=='A' else (x&yp,y&zp)
def mul(case,g,h):
    v,c=g;w,d=h;a,b=f(case,v,w)
    return v^w,c^d^a^(b<<1)
def commute(case,g,h):return mul(case,g,h)==mul(case,h,g)
def span(gens):
    vals={0}
    for g in gens:vals|={x^g for x in list(vals)}
    return sorted(vals)
def lift(vs):return {(v,c) for v in vs for c in range(4)}
def check_subgroup(case,H):return (0,0) in H and all(mul(case,a,b) in H for a in H for b in H)
def check_abelian(case,H):return all(commute(case,a,b) for a in H for b in H)
def check_clique(case,C):return all(not commute(case,C[i],C[j]) for i in range(len(C)) for j in range(i))
def caseA():
    xy=[1,2,3];zt=[4,8,12]
    return [(p^q,0) for p in xy for q in zt],[lift(span([p,q])) for p in xy for q in zt]
def caseB():
    clique=[(1,0)]+[(x|2|(z<<2),0) for x in (0,1) for z in (0,1)]
    covers=[lift(span([1,4,8]))]
    for x in (0,1):
     for z in (0,1):covers.append(lift(span([x|2|(z<<2),8])))
    return clique,covers

def run(case,builder):
    clique,covers=builder();universe={(v,c) for v in range(16) for c in range(4)}
    assert check_clique(case,clique)
    assert all(check_subgroup(case,H) and check_abelian(case,H) for H in covers)
    union=set().union(*covers);assert union==universe
    badcl=list(clique);badcl[-1]=badcl[0]
    assert not check_clique(case,badcl)
    assert set().union(*covers[:-1])!=universe
    return {'case':case,'omega_lower_witness_size':len(clique),'abelian_cover_upper_witness_size':len(covers),'clique':clique,'cover_sizes':[len(H) for H in covers],'full_group_covered':len(union)==64,'subgroups_closed':True,'subgroups_abelian':True,'negative_controls_passed':2,'conclusion':f'omega >= {len(clique)} and a <= {len(covers)}; since omega <= a for any group, omega=a={len(clique)}'}
out={'status':'PASS','method':'explicit full-group witnesses checked via raw multiplication only','cases':[run('A',caseA),run('B',caseB)]}
(OUT/'worker2_certificate.json').write_text(json.dumps(out,indent=2,default=list))
print(json.dumps(out,default=list))
