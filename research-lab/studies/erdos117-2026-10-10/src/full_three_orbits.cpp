// Exhaustive GL(5,2)-orbit reduction of 3-dimensional subspaces in
// Alt((F_2)^5); independently checks graph colorability/clique in each orbit.
// Prepared before worker 2. No literature novelty claimed.
#include <bits/stdc++.h>
using namespace std;
using U=uint32_t;
const U ALL31=0x7fffffffu;
static int wedge[32][32];
static unsigned short xform[2][1024];
static U neigh[31];
static int cliqueBest;
static int forbid[31],colorv[31],deg[31],K;
static unsigned long long colorsVisited;
static unsigned long long totalPlanes,orbitCount;
static unordered_set<U> seen;
static int hist[32]={};

inline U pack(int a,int b,int c){return U(a)|U(b)<<10|U(c)<<20;}
// Canonical reduced-row-echelon basis of a 3-plane in F_2^10.
U reduce3(int a,int b,int c){
 int rows[3]={a,b,c},rank=0;
 for(int i=0;i<10 && rank<3;i++){
  int piv=rank;while(piv<3 && !(rows[piv]&(1<<i)))piv++;
  if(piv==3)continue;
  swap(rows[piv],rows[rank]);
  for(int r=0;r<3;r++)if(r!=rank && (rows[r]&(1<<i)))rows[r]^=rows[rank];
  rank++;
 }
 if(rank!=3){cerr<<"RANK_LOST\n";exit(20);}
 return pack(rows[0],rows[1],rows[2]);
}
U apply(U s,int g){
 int a=s&1023,b=(s>>10)&1023,c=(s>>20)&1023;
 return reduce3(xform[g][a],xform[g][b],xform[g][c]);
}
static U matCols[2][5];
U lin(int g,U x){U out=0;for(int i=0;i<5;i++)if(x>>i&1)out^=matCols[g][i];return out;}
void precompute(){
 int id=0;
 for(int i=0;i<5;i++)for(int j=i+1;j<5;j++){
  for(int x=0;x<32;x++)for(int y=0;y<32;y++)
   if(((x>>i&1)&(y>>j&1))^((x>>j&1)&(y>>i&1)))wedge[x][y]|=1<<id;
  id++;
 }
 // g=0: five-cycle, g=1: elementary transvection.
 for(int i=0;i<5;i++){
  matCols[0][i]=1<<((i+1)%5);
  matCols[1][i]=1<<i;
 }
 matCols[1][1]^=1<<0;
 for(int g=0;g<2;g++)for(int m=0;m<1024;m++){
  int out=0,bit=0;
  for(int i=0;i<5;i++)for(int j=i+1;j<5;j++){
   if(__builtin_parity((U)(m&wedge[matCols[g][i]][matCols[g][j]])))out|=1<<bit;
   ++bit;
  }
  xform[g][m]=out;
 }
 for(int m=0;m<1024;m++){
  int v=m;for(int i=0;i<5;i++)v=xform[0][v];
  if(v!=m || xform[1][xform[1][m]]!=m){cerr<<"BAD_GENERATOR_ORDER\n";exit(21);}
 }
}

void colorSort(U P,int order[31],int colors[31],int &n){
 n=0;int color=0;
 while(P){color++;U Q=P;
  while(Q){int v=__builtin_ctz(Q);U bit=1u<<v;order[n]=v;colors[n]=color;n++;
   P&=~bit;Q&=~(bit|neigh[v]);}}
}
void clique(U P,int depth){
 if(!P){cliqueBest=max(cliqueBest,depth);return;}
 int order[31],colors[31],n=0;colorSort(P,order,colors,n);
 for(int i=n-1;i>=0;i--){if(depth+colors[i]<=cliqueBest)return;
  int v=order[i];clique(P&neigh[v],depth+1);P&=~(1u<<v);}
}
bool recColor(int assigned,int used){
 colorsVisited++;
 if(assigned==31)return true;
 int v=-1,maxSat=-1,maxDeg=-1;
 for(int i=0;i<31;i++)if(colorv[i]<0){int sat=__builtin_popcount((U)forbid[i]);
  if(sat>maxSat ||(sat==maxSat&&deg[i]>maxDeg))v=i,maxSat=sat,maxDeg=deg[i];}
 U avail=((1u<<min(used,K))-1u)&~(U)forbid[v];
 while(avail){int c=__builtin_ctz(avail);avail&=avail-1;
  colorv[v]=c;int changed[31],q=0;U n=neigh[v];
  while(n){int j=__builtin_ctz(n);n&=n-1;if(colorv[j]<0&&!(forbid[j]&(1<<c))){forbid[j]|=1<<c;changed[q++]=j;}}
  if(recColor(assigned+1,used))return true;
  for(int z=0;z<q;z++)forbid[changed[z]]&=~(1<<c);
  colorv[v]=-1;
 }
 if(used<K){int c=used;colorv[v]=c;int changed[31],q=0;U n=neigh[v];
  while(n){int j=__builtin_ctz(n);n&=n-1;if(colorv[j]<0&&!(forbid[j]&(1<<c))){forbid[j]|=1<<c;changed[q++]=j;}}
  if(recColor(assigned+1,used+1))return true;
  for(int z=0;z<q;z++)forbid[changed[z]]&=~(1<<c);
  colorv[v]=-1;
 }
 return false;
}
bool colorable(int k){
 K=k;colorsVisited=0;
 for(int i=0;i<31;i++)colorv[i]=-1,forbid[i]=0,deg[i]=__builtin_popcount(neigh[i]);
 return recColor(0,0);
}

// Independently reconstruct vector adjacency from symmetric matrices.
// Also verify both change-of-basis generator actions on all 31x31 pairs.
int eval(int form,U x,U y){
 int M[5][5]={};int z=0;
 for(int i=0;i<5;i++)for(int j=i+1;j<5;j++){M[i][j]=M[j][i]=(form>>z)&1;z++;}
 int ans=0;
 for(int i=0;i<5;i++)if(x>>i&1)for(int j=0;j<5;j++)if(y>>j&1)ans^=M[i][j];
 return ans;
}
int graph(int a,int b,int c){
 int edges=0;
 for(int x=1;x<32;x++){neigh[x-1]=0;
  for(int y=1;y<32;y++)if(x!=y){int z=(bool)eval(a,x,y)||(bool)eval(b,x,y)||(bool)eval(c,x,y);
   int w=wedge[x][y];
   int independent=(bool)(__builtin_parity(U(a&w))||__builtin_parity(U(b&w))||__builtin_parity(U(c&w)));
   if(z!=independent){cerr<<"GRAPH_REPRESENTATION_MISMATCH\n";exit(22);}
   if(z)neigh[x-1]|=1u<<(y-1);
  }
  edges+=__builtin_popcount(neigh[x-1]);
 }
 return edges/2;
}
void checkGenerators(U s){
 int a=s&1023,b=(s>>10)&1023,c=(s>>20)&1023;
 for(int g=0;g<2;g++){
  int a2=xform[g][a],b2=xform[g][b],c2=xform[g][c];
  for(int x=0;x<32;x++)for(int y=0;y<32;y++){
   U px=lin(g,x),py=lin(g,y);
   bool f=(bool)(eval(a,px,py)||eval(b,px,py)||eval(c,px,py));
   bool t=(bool)(eval(a2,x,y)||eval(b2,x,y)||eval(c2,x,y));
   if(f!=t){cerr<<"CHANGE_OF_BASIS_MISMATCH\n";exit(23);}
  }
 }
}
void analyze(U s,long long size){
 int a=s&1023,b=(s>>10)&1023,c=(s>>20)&1023;
 checkGenerators(s);int ed=graph(a,b,c);
 cliqueBest=0;clique(ALL31,0);int w=cliqueBest;
 bool k=colorable(w);
 if(!k){
  cout<<"GAP_ORBIT a="<<a<<" b="<<b<<" c="<<c<<" omega="<<w<<" orbit_size="<<size;
  bool upper=colorable(w+1);cout<<" omega_plus_one="<<upper<<" edges="<<ed<<"\n";
  if(upper){cout<<"UPPER_COLORING";for(int i=0;i<31;i++)cout<<' '<<colorv[i];cout<<"\n";}
  cout<<flush; exit(25);
 }
 for(int i=0;i<31;i++)for(int j=0;j<31;j++)
  if((neigh[i]>>j&1)&&colorv[i]==colorv[j]){cerr<<"INVALID_COVER_COLORING\n";exit(26);}
 hist[w]++;
 cout<<"ORBIT "<<orbitCount<<" size="<<size<<" basis="<<a<<","<<b<<","<<c<<" omega=a="<<w<<" edges="<<ed<<" total="<<totalPlanes<<"\n"<<flush;
}
int main(){
 precompute();
 // Unit smoke: RREF orbit basis canonicalization, the zero graph, generators.
 if(reduce3(1,2,4)!=pack(1,2,4)){cerr<<"RREF_SMOKE_FAIL\n";return 8;}
 if(graph(0,0,0)!=0){cerr<<"ZERO_GRAPH_FAIL\n";return 9;}
 cliqueBest=0;clique(ALL31,0);
 if(cliqueBest!=1||!colorable(1)){cerr<<"ZERO_COLOR_FAIL\n";return 10;}
 cout<<"SMOKE_OK generator_order5_order2, zero_graph, rref\n"<<flush;
 seen.max_load_factor(0.77);seen.reserve(6400000);
 for(int p=0;p<8;p++)for(int q=p+1;q<9;q++)for(int r=q+1;r<10;r++){
  U piv=(1u<<p)|(1u<<q)|(1u<<r);
  U m0=0,m1=0,m2=0;
  for(int z=0;z<10;z++)if(!(piv>>z&1)){
   if(z>p)m0|=1u<<z;
   if(z>q)m1|=1u<<z;
   if(z>r)m2|=1u<<z;
  }
  // enumerate all submasks, including zero.
  for(U sa=m0;;sa=(sa-1)&m0){
   for(U sb=m1;;sb=(sb-1)&m1){
    for(U sc=m2;;sc=(sc-1)&m2){
     U key=pack((1u<<p)|sa,(1u<<q)|sb,(1u<<r)|sc);
     if(seen.insert(key).second){
      orbitCount++;
      vector<U> Q;Q.push_back(key);
      for(size_t t=0;t<Q.size();t++){
       U s=Q[t];
       for(int g=0;g<2;g++){
        U next=apply(s,g);
        if(seen.insert(next).second)Q.push_back(next);
       }
      }
      totalPlanes+=Q.size();
      analyze(key,Q.size());
     }
     if(sc==0)break;
    }
    if(sb==0)break;
   }
   if(sa==0)break;
  }
 }
 const unsigned long long expected=6347715;
 cout<<"COMPLETE_RANK5_THREE_PLANE_CENSUS visited="<<seen.size()<<" total="<<totalPlanes<<" orbits="<<orbitCount<<" expected="<<expected<<"\n";
 cout<<"ORBIT_OMEGA_HIST";for(int w=0;w<32;w++)if(hist[w])cout<<" "<<w<<":"<<hist[w];cout<<"\n"<<flush;
 return (seen.size()==expected && totalPlanes==expected)?0:27;
}
