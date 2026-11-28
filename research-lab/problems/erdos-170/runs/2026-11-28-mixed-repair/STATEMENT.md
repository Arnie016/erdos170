# Frozen target: one strict middle mark and one exterior mark

Let $r\geq 1$ be an integer. Put $s=2r$, $a=2r+1$, $b=2r+2$, $q=a+b$, $L=(r+1)a$, and $N=L+sq+r+(r+1)b$. Define the five sets of integer marks

$$
\begin{aligned}
B_0 &= \{0,1,\ldots,r\},\\
B_1 &= \{ia:1\leq i\leq r\},\\
B_2 &= \{L+jq:0\leq j\leq s\},\\
B_3 &= \{N-r-ib:1\leq i\leq r\},\\
B_4 &= \{N-r,\ldots,N\}.
\end{aligned}
$$

Write $W=\bigcup_{i=0}^4 B_i$. Choose $u=L+jq$ for an integer $1\leq j\leq s-1$, choose $v\in B_0\cup B_1\cup B_3\cup B_4$, and choose an integer $z\in[0,N]\setminus W$. Set $A=(W\setminus\{u,v\})\cup\{z\}$. Freshness is relative to the original $W$, so a deleted mark cannot be reinserted. An exterior choice includes either original endpoint. The target length remains $N$, and coverage of $N$ forces both original endpoints to survive. The two junctions $L,L+sq$ are in $B_2$ and cannot be deleted in this target.

**Question.** Do there exist such $r,j,v,z$ with $\Delta^+(A)=\{1,\ldots,N\}$, where $\Delta^+(A)=\{y-x:x,y\in A,\ x<y\}$?

The proposed obstruction is the universal negative of this question.
