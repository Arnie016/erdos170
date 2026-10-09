#include <bits/stdc++.h>
using namespace std;
using U=uint32_t;
static U adj[31];
static int wedge[32][32];
int bestClique;
int color_calls;
void color_sort(U P, int order[31], int colors[31],int& n){
 n=0; int col=0;
 while(P){col++; U Q=P; while(Q){int v=__builtin_ctz(Q);U bit=U(1)<<v; order[n]=v;colors[n]=col;n++; P&=~bit;Q&=~(bit|adj[v]);}}
}
void clique_search(U P,int depth){
 if(!P){bestClique=max(bestClique,depth);return;}
 int order[31],colors[31],n=0;color_sort(P,order,colors,n);
 for(int i=n-1;i>=0;i--){if(depth+colors[i]<=bestClique)return;int v=order[i];clique_search(P&adj[v],depth+1);P&=~(U(1)<<v);}
}
// Exact graph k-colourability by vertex-saturation backtracking.
static int forbidMask[31], vertexColor[31], degreeArr[31], targetColors;
static unsigned long long nodes;
bool try_k_colors(int coloured, int used){
 if(coloured==31)return true;
 int v=-1,bestSat=-1,bestDegree=-1;
 for(int i=0;i<31;i++) if(vertexColor[i]<0){int sat=__builtin_popcount((unsigned)forbidMask[i]);if(sat>bestSat ||(sat==bestSat&&degreeArr[i]>bestDegree)){v=i;bestSat=sat;bestDegree=degreeArr[i];}}
 U available=(((U(1)<<min(used,targetColors))-1))&~U(forbidMask[v]);
 while(available){int c=__builtin_ctz(available);available&=available-1;
  vertexColor[v]=c;int changed[31],q=0;U neigh=adj[v];while(neigh){int j=__builtin_ctz(neigh);neigh&=neigh-1;if(vertexColor[j]<0&&!(forbidMask[j]&(1<<c))){forbidMask[j]|=1<<c;changed[q++]=j;}}
  if(try_k_colors(coloured+1,used))return true;
  for(int x=0;x<q;x++) forbidMask[changed[x]] &= ~(1<<c);vertexColor[v]=-1;
 }
 if(used<targetColors){int c=used;vertexColor[v]=c;int changed[31],q=0;U neigh=adj[v];while(neigh){int j=__builtin_ctz(neigh);neigh&=neigh-1;if(vertexColor[j]<0&&!(forbidMask[j]&(1<<c))){forbidMask[j]|=1<<c;changed[q++]=j;}}
  if(try_k_colors(coloured+1,used+1))return true;
  for(int x=0;x<q;x++) forbidMask[changed[x]] &= ~(1<<c);vertexColor[v]=-1;
 }
 return false;
}
int analyse(int a,int b,bool& kColor){
 for(int i=1;i<32;i++){adj[i-1]=0;for(int j=1;j<32;j++) if(i!=j){int w=wedge[i][j];if((__builtin_parity((unsigned)(a&w)))||(__builtin_parity((unsigned)(b&w))))adj[i-1]|=U(1)<<(j-1);}}
 bestClique=0;clique_search((U(1)<<31)-1,0);
 for(int i=0;i<31;i++){forbidMask[i]=0;vertexColor[i]=-1;degreeArr[i]=__builtin_popcount(adj[i]);}
 targetColors=bestClique;nodes=0;kColor=try_k_colors(0,0);return bestClique;
}
int main(int argc,char**argv){
 int pos=0;for(int i=0;i<5;i++)for(int j=i+1;j<5;j++){for(int u=0;u<32;u++)for(int v=0;v<32;v++){if((u>>i&1) && (v>>j&1))wedge[u][v]^=1<<pos; if((u>>j&1)&&(v>>i&1))wedge[u][v]^=1<<pos;}pos++;}
 bool c;int om=analyse(0,0,c);if(om!=1||!c){cerr<<"SMOKE_FAIL_ZERO\n";return 2;}
 om=analyse(1|8,0,c);if(om<3||!c){cerr<<"SMOKE_FAIL_SCALAR\n";return 2;}
 ofstream records("results/rank5_pencils.jsonl");if(!records){cerr<<"OUTPUT_FAIL\n";return 3;}
 unsigned long long checked=0,gaps=0;int hist[32]={};
 for(int i=0;i<10;i++)for(int j=i+1;j<10;j++){
  vector<int> q1,q2;for(int k=i+1;k<10;k++)if(k!=j)q1.push_back(k);for(int k=j+1;k<10;k++)q2.push_back(k);
  for(int p=0;p<(1<<q1.size());p++) for(int q=0;q<(1<<q2.size());q++){
   int a=1<<i,b=1<<j;for(int k=0;k<(int)q1.size();k++)if(p>>k&1)a|=1<<q1[k];for(int k=0;k<(int)q2.size();k++)if(q>>k&1)b|=1<<q2[k];
   bool kColor=false;int omega=analyse(a,b,kColor);++checked;hist[omega]++;
   records<<"{\"a\":"<<a<<",\"b\":"<<b<<",\"omega\":"<<omega<<",\"colorable\":"<<(kColor?"true":"false")<<"}\n";
   if(!kColor){gaps++; records.flush();cout<<"{\"status\":\"GAP_CANDIDATE\",\"a\":"<<a<<",\"b\":"<<b<<",\"omega\":"<<omega<<",\"checked\":"<<checked<<"}"<<endl;return 0;}
   if(checked%2000==0){records.flush();cerr<<"checkpoint "<<checked<<"\n";}
  }
 }
 cerr<<"full tested: "<<checked<<"\n";
 cout<<"{\"status\":\"FULL_NO_GAP\",\"checked\":"<<checked<<",\"gaps\":"<<gaps<<",\"histogram\":{";
 for(int k=0;k<32;k++)if(hist[k])cout<<"\""<<k<<"\":"<<hist[k]<<",";cout<<"\"end\":0}}"<<endl;
 return checked==174251?0:4;
}
