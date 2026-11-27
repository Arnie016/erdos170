#!/usr/bin/env python3
"""Independent Sylvester resultant of F_d(X) and P(X), integer determinant."""
import sys,json,hashlib,time
from pathlib import Path
TARGET=[1,0,0,1,1,1,0,0,1]
POS=[1,0,1]
NEG=[1]*11

def bareiss(A):
    a=[list(row) for row in A];n=len(a);sgn=1;den=1
    for j in range(n-1):
        pivot=None
        for i in range(j,n):
            if a[i][j]:pivot=i;break
        if pivot is None:return 0
        if pivot!=j:a[j],a[pivot]=a[pivot],a[j];sgn=-sgn
        x=a[j][j]
        for i in range(j+1,n):
            for k in range(j+1,n):
                v=x*a[i][k]-a[i][j]*a[j][k]
                assert v%den==0,(j,i,k,den,v)
                a[i][k]=v//den
            a[i][j]=0
        den=x
    return sgn*a[n-1][n-1]

def resultant(d,P):
    # Matrix columns represent a(X)F_d(X)+b(X)P(X), where deg a<deg P,
    # deg b<deg F_d; rows are coefficient equations in ascending degree.
    F=[1]*d
    m=len(F)-1;n=len(P)-1
    size=m+n
    M=[[0]*size for _ in range(size)]
    for c in range(n):
        for k,x in enumerate(F):M[c+k][c]=x
    for c in range(m):
        for k,x in enumerate(P):M[c+k][n+c]=x
    # Known Sylvester column permutation has sign (-1)^(m*n), both even P degree
    # in all declared cases; no further sign adjustment needed.
    assert (m*n)%2==0
    return bareiss(M)
start=time.perf_counter()
out={str(d):str(resultant(d,TARGET)) for d in (11,47)}
checks={'positive_11':str(resultant(11,POS)),'positive_47':str(resultant(47,POS)), 'zero_11':str(resultant(11,NEG))}
assert checks=={'positive_11':'1','positive_47':'1','zero_11':'0'}, checks
receipt={'method':'direct Sylvester integer determinant', 'resultants':out, 'controls':checks,'internal_wall_seconds':time.perf_counter()-start,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(sys.argv[1]).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
