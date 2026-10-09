import json, math, time, sys, hashlib
from pathlib import Path
started=time.monotonic()
source=Path(sys.argv[1]); outpath=Path(sys.argv[2])
data=json.loads(source.read_text())
assert data["N"]==69 and data["n"]==6
N=69; n=6; h=34
reconstructed=[]; visited=[0]*(n+1); failures=[]
def build(A,residues,start):
    depth=len(A); visited[depth]+=1
    if depth==n:
        reconstructed.append(tuple(A)); return
    for a in range(start,h-(n-depth)+2):
        shifted={(r+a)%N for r in residues}
        if residues.isdisjoint(shifted):
            build(A+(a,),residues|shifted,a+1)
build((),{0},1)
got=sorted(tuple(r["canonical_set"]) for r in data["records"])
assert len(got)==len(set(got))
assert got==sorted(reconstructed)
assert data["canonical_count"]==len(got)
assert data["full_count"]==len(got)*64
assert data["domain"]["canonical_full_combinations"]==math.comb(34,6)
assert data["domain"]["original_full_combinations"]==math.comb(68,6)

signed_orbits=set()
for u in range(1,N):
    if math.gcd(u,N)!=1: continue
    dyadic=[(u*pow(2,i,N))%N for i in range(n)]
    for signs in range(1<<n):
        B=tuple(sorted(((-x)%N if (signs>>i)&1 else x) for i,x in enumerate(dyadic)))
        assert len(set(B))==n
        signed_orbits.add(B)
orbitsets=[set(B) for B in signed_orbits]

def direct_subset_sums(A):
    return [sum(A[i] for i in range(n) if (mask>>i)&1)%N for mask in range(1<<n)]
hist={str(i):0 for i in range(n+1)}
all_lifts=set(); subset_evaluations=0; minimum_overlap=n
for r in data["records"]:
    A=tuple(r["canonical_set"])
    sums=direct_subset_sums(A);subset_evaluations+=len(sums)
    assert len(set(sums))==64
    overlap=max(len(set(A)&B) for B in orbitsets)
    assert overlap==r["best_overlap"]
    assert overlap>=5
    hist[str(overlap)]+=1
    minimum_overlap=min(minimum_overlap,overlap)
    u=r["unit_witness"]; assert math.gcd(u,N)==1
    canonical=sorted(min((u*pow(2,i,N))%N,N-(u*pow(2,i,N))%N) for i in range(n))
    assert canonical==r["canonical_dyadic"]
    assert sorted(set(A)&set(canonical))==r["intersection"]
    for signs in range(64):
        lift=tuple(sorted(((-a)%N if (signs>>i)&1 else a) for i,a in enumerate(A)))
        assert len(set(lift))==6
        lsums=direct_subset_sums(lift);subset_evaluations+=64
        assert len(set(lsums))==64
        all_lifts.add(lift)
assert len(all_lifts)==len(got)*64
assert data["overlap_histogram"]==hist
negative_controls={}
removed=got[:-1]
negative_controls["missing_valid_canonical_set_rejected"]=(removed!=sorted(reconstructed))
fake=(1,2,3,4,5,6)
negative_controls["colliding_subset_sum_set_rejected"]=(len(set(direct_subset_sums(fake)))!=64)
assert all(negative_controls.values())
out={"status":"EXACT_FINITE_CERTIFICATE","n":n,"N":N,
     "canonical_count":len(got),"full_count":len(all_lifts),
     "minimum_overlap":minimum_overlap,"overlap_histogram":hist,
     "nodes_by_depth":visited,"explicit_signed_dyadic_orbits":len(signed_orbits),
     "direct_subset_sum_values_checked":subset_evaluations,
     "negative_controls":negative_controls,
     "all_n_theorem":False,"novelty":"PRIORITY_UNCHECKED",
     "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
     "elapsed_seconds":time.monotonic()-started}
outpath.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
