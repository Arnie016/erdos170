// Exact rank-5 3-output random linear-map search; frozen sample seed and parameters.
// Domain: first 1000 distinct independent scalar triples with zero common radical from xorshift32 seed 0x1172026.
// Checker: exact Bron-Kerbosch clique number, DSATUR k-colorability with backtracking; candidate certificate is 10-bit triple.
#include <bits/stdc++.h>
using namespace std;
using ull = uint64_t;
static inline int pc(ull x){return __builtin_popcountll(x);}
static uint32_t s = 0x1172026u;
uint32_t rand32(){s ^=s<<13; s^=s>>17;s^=s<<5;return s;}
int pairs[5][5], wedge[32][32];
struct Graph {
 ull adj[31]={}; int degree[31]={};
 int bestclique=0; ull bestmask=0;
 bool exhausted=true; long long calls=0;
 void bk(int size, ull mask, ull P, ull X){
  if(P==0 && X==0){if(size>bestclique){bestclique=size;bestmask=mask;}return;}
  if(size+pc(P)<=bestclique)return;
  ull U=P|X,chosen=0; int best=-1;
  while(U){int u=__builtin_ctzll(U);U&=U-1;int v=pc(P&adj[u]);if(v>best){best=v;chosen=adj[u];}}
  ull possible=P & ~chosen;
  while(possible){int v=__builtin_ctzll(possible);ull bit=1ULL<<v;possible-=bit;
   bk(size+1,mask|bit,P&adj[v],X&adj[v]);P&=~bit;X|=bit;
   if(size+pc(P)<=bestclique)return;
  }
 }
 int greedy(){
  int clr[31];fill(clr,clr+31,-1);int maxc=-1;ull U=(1ULL<<31)-1;
  while(U){int choice=-1,bestsat=-1,bestdeg=-1;
    for(ull v=U;v;v&=v-1){int x=__builtin_ctzll(v);ull used=0;for(int y=0;y<31;y++)if(clr[y]>=0&&(adj[x]>>y&1))used|=1ULL<<clr[y];int sat=pc(used);
       if(sat>bestsat||(sat==bestsat&&degree[x]>bestdeg)){choice=x;bestsat=sat;bestdeg=degree[x];}}
    ull used=0;for(int y=0;y<31;y++)if(clr[y]>=0&&(adj[choice]>>y&1))used|=1ULL<<clr[y];int c=0;while(used>>c&1)c++;
    clr[choice]=c;maxc=max(maxc,c);U&=~(1ULL<<choice);
  }
  return maxc+1;
 }
 bool dfs(ull U, int maxc, int k, int color[31], int depth){
  if(!U)return true;
  if(++calls>3000000){exhausted=false;return false;}
  int choice=-1,bestsat=-1,bestdeg=-1;ull usedChoice=0;
  for(ull v=U;v;v&=v-1){int x=__builtin_ctzll(v);ull used=0;
    ull W=adj[x]&~U;while(W){int y=__builtin_ctzll(W);W&=W-1;used|=1ULL<<color[y];}
    int sat=pc(used);
    if(sat>bestsat||(sat==bestsat&&degree[x]>bestdeg)){choice=x;bestsat=sat;bestdeg=degree[x];usedChoice=used;}
  }
  for(int c=0;c<=min(maxc+1,k-1);c++) if(!(usedChoice>>c&1)){
    color[choice]=c;
    if(dfs(U&~(1ULL<<choice),max(maxc,c),k,color,depth+1))return true;
    color[choice]=-1;
    if(!exhausted)return false;
  }
  return false;
 }
 bool colorable(int k){int color[31];fill(color,color+31,-1);calls=0;exhausted=true;return dfs((1ULL<<31)-1,-1,k,color,0);}
};
int main(){auto t0=chrono::steady_clock::now();int p=0;for(int i=0;i<5;i++)for(int j=i+1;j<5;j++)pairs[i][j]=p++;
 for(int x=0;x<32;x++)for(int y=0;y<32;y++){
  int w=0;for(int i=0;i<5;i++)for(int j=i+1;j<5;j++)if(((x>>i&1)&(y>>j&1))^((x>>j&1)&(y>>i&1)))w|=1<<pairs[i][j];wedge[x][y]=w;
 }
 int trials=0,accepted=0,eq=0,unknown=0;set<array<int,3>>seen;
 while(accepted<1000 && trials<50000){trials++;array<int,3>a={(int)(rand32()&1023),(int)(rand32()&1023),(int)(rand32()&1023)};sort(a.begin(),a.end());
  if(!a[0]||a[0]==a[1]||a[1]==a[2]||a[0]==(a[1]^a[2])||!seen.insert(a).second)continue;
  Graph g;bool hasrad=false;
  for(int x=1;x<32;x++){
    bool allzero=true;
    for(int y=1;y<32;y++){
      int w=wedge[x][y];bool edge= (__builtin_parity((unsigned)(w&a[0])) | __builtin_parity((unsigned)(w&a[1])) | __builtin_parity((unsigned)(w&a[2])));
      if(edge) allzero=false;
      if(x!=y&&edge) g.adj[x-1]|=1ULL<<(y-1);
    }
    if(allzero){hasrad=true;break;}
  }
  if(hasrad)continue;
  accepted++;
  for(int i=0;i<31;i++)g.degree[i]=pc(g.adj[i]);
  g.bk(0,0,(1ULL<<31)-1,0);
  int greedy=g.greedy();
  if(greedy<=g.bestclique){eq++;continue;}
  bool can=g.colorable(g.bestclique);
  if(!g.exhausted){unknown++;continue;}
  if(!can){auto secs=chrono::duration<double>(chrono::steady_clock::now()-t0).count();
   cout<<"GAP triple="<<a[0]<<","<<a[1]<<","<<a[2]<<" omega="<<g.bestclique<<" greedy="<<greedy<<" chromatic_at_least="<<g.bestclique+1<<" accepted_index="<<accepted<<" trials="<<trials<<" color_calls="<<g.calls<<" elapsed="<<secs<<"\n";
   for(int i=0;i<31;i++)cout<<i+1<<" "<<g.adj[i]<<"\n";
   return 0;}
  eq++;
 }
 auto secs=chrono::duration<double>(chrono::steady_clock::now()-t0).count();
 cout<<"NO_WITNESS accepted="<<accepted<<" eq="<<eq<<" unknown="<<unknown<<" total_trials="<<trials<<" elapsed="<<secs<<"\n";
}
