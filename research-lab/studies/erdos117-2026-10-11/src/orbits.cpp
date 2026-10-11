// October 11, 2026 research unit: enumerate all 3-dimensional scalar-form subspaces
// P <= Alt(F_2^5) under GL(5,2), determine whether chi(noncommutation)>omega.
// This is WORKER 2 (the final permitted daily worker). Exact integer bit operations.
// Predeclared acceptance: 6,347,715 RREF planes, all visited by orbit BFS; every
// orbit representative receives exact clique, and chi<=omega is checked by explicit
// greedy coloring or exhaustive DSATUR. If 35s timeout, only partial orbits stand.
#include <bits/stdc++.h>
using namespace std;
using U=uint32_t;using L=uint64_t;
static inline int pc(L x){return __builtin_popcountll(x);}
static inline int parity(int x){return __builtin_parity((unsigned)x);}
int pairnum[5][5],wedge[32][32],maps[5][1024];
U reduce3(int a,int b,int c){
 int x[3]={a,b,c};int p=0;
 for(int col=0;col<10&&p<3;col++){
  int r=p;while(r<3&&!(x[r]&(1<<col)))r++;
  if(r==3)continue;
  swap(x[p],x[r]);
  for(int j=0;j<3;j++)if(j!=p&&(x[j]&(1<<col)))x[j]^=x[p];
  p++;
 }
 if(p!=3){cerr<<"BAD_RANK\n";exit(3);}
 return (U)x[0]|((U)x[1]<<10)|((U)x[2]<<20);
}
void init(){int k=0;for(int i=0;i<5;i++)for(int j=i+1;j<5;j++)pairnum[i][j]=k++;
 for(int x=0;x<32;x++)for(int y=0;y<32;y++){
  int w=0;for(int i=0;i<5;i++)for(int j=i+1;j<5;j++)if(((x>>i&1)&(y>>j&1))^((x>>j&1)&(y>>i&1)))w|=1<<pairnum[i][j];
  wedge[x][y]=w;
 }
 for(int t=0;t<5;t++){
  int cols[5];for(int i=0;i<5;i++)cols[i]=1<<i;
  if(t<4)swap(cols[t],cols[t+1]);else cols[0]^=cols[1];
  for(int a=0;a<1024;a++){
   int transformed=0;
   for(int i=0;i<5;i++)for(int j=i+1;j<5;j++)if(parity(a&wedge[cols[i]][cols[j]]))transformed|=1<<pairnum[i][j];
   maps[t][a]=transformed;
  }
  for(int a=0;a<1024;a++)if(maps[t][maps[t][a]]!=a){cerr<<"FAIL_INVOLUTION gen="<<t<<" form="<<a<<"\n";exit(2);}
 }
 if(reduce3(1,2,4)!=(1U|(2U<<10)|(4U<<20))){cerr<<"FAIL_RREF_SMOKE\n";exit(2);}
 cout<<"SMOKE_PASS involutions_5x1024 and RREF_identity\n";
}
struct Graph{
 L adj[31]={};int deg[31]={};int omega=0;long long color_nodes=0;bool complete=true;
 void BK(int r,L P,L X){
  if(!P&&!X){omega=max(omega,r);return;}
  if(r+pc(P)<=omega)return;
  L S=P|X,piv=0;int best=-1;
  while(S){int v=__builtin_ctzll(S);S&=S-1;int c=pc(P&adj[v]);if(c>best){best=c;piv=adj[v];}}
  L opts=P&~piv;
  while(opts){int v=__builtin_ctzll(opts);L bit=1ULL<<v;opts-=bit;
   BK(r+1,P&adj[v],X&adj[v]);P&=~bit;X|=bit;
   if(r+pc(P)<=omega)return;
  }
 }
 int greedyColor(){
  int assigned[31];fill(assigned,assigned+31,-1);L U=(1ULL<<31)-1;int k=0;
  while(U){int chosen=-1,bestsat=-1,bestdeg=-1;L chosenused=0;
   for(L Q=U;Q;Q&=Q-1){int v=__builtin_ctzll(Q);L taken=0;
    for(L Z=adj[v]&~U;Z;Z&=Z-1){int j=__builtin_ctzll(Z);taken|=1ULL<<assigned[j];}
    int sat=pc(taken);if(sat>bestsat||(sat==bestsat&&deg[v]>bestdeg)){
      chosen=v;bestsat=sat;bestdeg=deg[v];chosenused=taken;
    }
   }
   int col=0;while(chosenused>>col&1)col++;k=max(k,col+1);assigned[chosen]=col;U&=~(1ULL<<chosen);
  }return k;
 }
 bool dsatur(L U,int colorsUsed,int k,int assigned[31]){
  if(!U)return true;
  if(++color_nodes>3000000){complete=false;return false;}
  int chosen=-1,bestsat=-1,bestdeg=-1;L chosenused=0;
  for(L Q=U;Q;Q&=Q-1){int v=__builtin_ctzll(Q);L taken=0;
   for(L Z=adj[v]&~U;Z;Z&=Z-1){int j=__builtin_ctzll(Z);taken|=1ULL<<assigned[j];}
   int sat=pc(taken);if(sat>bestsat||(sat==bestsat&&deg[v]>bestdeg)){
    chosen=v;bestsat=sat;bestdeg=deg[v];chosenused=taken;
   }
  }
  for(int col=0;col<=min(colorsUsed,k-1);col++)if(!(chosenused>>col&1)){
   assigned[chosen]=col;
   if(dsatur(U&~(1ULL<<chosen),max(colorsUsed,col+1),k,assigned))return true;
   assigned[chosen]=-1;
   if(!complete)return false;
  }
  return false;
 }
 bool colorable(int k){int a[31];fill(a,a+31,-1);return dsatur((1ULL<<31)-1,0,k,a);}
};
struct Outcome {int omega,greedy,rad;bool gap,unknown;long long searchNodes;};
Outcome check(U key){int forms[3]={int(key&1023),int((key>>10)&1023),int((key>>20)&1023)};
 Graph g;int rad=0;
 for(int x=1;x<32;x++)for(int y=x+1;y<32;y++){
  int w=wedge[x][y];bool edge=parity(w&forms[0])||parity(w&forms[1])||parity(w&forms[2]);
  if(edge){g.adj[x-1]|=1ULL<<(y-1);g.adj[y-1]|=1ULL<<(x-1);}
 }
 for(int i=0;i<31;i++){g.deg[i]=pc(g.adj[i]);if(g.deg[i]==0)rad++;}
 g.BK(0,(1ULL<<31)-1,0);
 int greedy=g.greedyColor();
 if(greedy==g.omega)return {g.omega,greedy,rad,false,false,0};
 bool feasible=g.colorable(g.omega);
 return {g.omega,greedy,rad,!feasible&&g.complete,!g.complete,g.color_nodes};
}
int main(){ios::sync_with_stdio(false);cout.setf(std::ios::unitbuf);init();auto now=chrono::steady_clock::now();
 constexpr U MAXKEY=1U<<30;
 vector<L> seen((size_t)MAXKEY/64,0);
 auto seenMark=[&](U key)->bool {L &cell=seen[key>>6];L bit=1ULL<<(key&63);if(cell&bit)return false;cell|=bit;return true;};
 vector<U> q;q.reserve(3000000);
 uint64_t total=0,orbits=0,degenerate=0,faithful=0,possibleGaps=0,unknown=0,greedyNotEnough=0;
 map<pair<int,int>,uint64_t> histogram;
 const uint64_t expected=6347715;
 for(int p0=0;p0<8;p0++)for(int p1=p0+1;p1<9;p1++)for(int p2=p1+1;p2<10;p2++){
  int pivot[3]={p0,p1,p2};
  vector<pair<int,int>> cells;
  U initial=0;
  for(int r=0;r<3;r++){
   initial|=(U)(1<<pivot[r])<<(10*r);
   for(int j=pivot[r]+1;j<10;j++)if(j!=p0&&j!=p1&&j!=p2)cells.push_back({r,j});
  }
  int N=cells.size();U key=initial,gray=0;
  for(int id=0;id<(1<<N);id++){
   if(id){int bit=__builtin_ctz((unsigned)id);key^= (U)(1<<cells[bit].second)<<(10*cells[bit].first);}
   if(!seenMark(key))continue;
   q.clear();q.push_back(key);orbits++;
   for(size_t pos=0;pos<q.size();pos++){
     U item=q[pos];int a=item&1023,b=(item>>10)&1023,c=(item>>20)&1023;
     for(int g=0;g<5;g++){
       U next=reduce3(maps[g][a],maps[g][b],maps[g][c]);
       if(seenMark(next))q.push_back(next);
     }
   }
   total+=q.size();
   Outcome out=check(key);
   if(out.rad)degenerate++ ;else faithful++;
   if(out.unknown)unknown++;
   if(out.greedy>out.omega)greedyNotEnough++;
   histogram[{out.omega,out.greedy}]+=q.size();
   cout<<"ORBIT root="<<key<<" forms="<<(key&1023)<<","<<((key>>10)&1023)<<","<<((key>>20)&1023)
       <<" size="<<q.size()<<" omega="<<out.omega<<" greedy="<<out.greedy
       <<" radical_vectors="<<out.rad<<" gap="<<out.gap<<" unknown="<<out.unknown<<"\n";
   if(out.gap){possibleGaps++;cout<<"GAP_CERT root="<<key<<" omega="<<out.omega<<" chi_lower="<<out.omega+1<<" nodes="<<out.searchNodes<<"\n";}
   if(orbits%100==0)cout<<"PROGRESS orbits="<<orbits<<" visited="<<total<<" elapsed="<<chrono::duration<double>(chrono::steady_clock::now()-now).count()<<"\n";
  }
 }
 cout<<"FINAL total="<<total<<" expected="<<expected<<" orbits="<<orbits<<" faithful_orbits="<<faithful<<" degenerate_orbits="<<degenerate<<" gaps="<<possibleGaps<<" unknown="<<unknown<<" greedy_bigger="<<greedyNotEnough<<" seconds="<<chrono::duration<double>(chrono::steady_clock::now()-now).count()<<"\n";
 for(auto [pair,count]:histogram)cout<<"HIST omega="<<pair.first<<" greedy="<<pair.second<<" planes="<<count<<"\n";
 if(total!=expected||unknown)return 2;
 return 0;
}
