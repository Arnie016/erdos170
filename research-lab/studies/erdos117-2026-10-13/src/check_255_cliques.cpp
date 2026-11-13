// Exhaustive exact clique reconstruction. C++17, no third-party dependencies.
// Mathematical domain: 255 cosets of span(129,66) in Alt(F_2^5), graph V\\{0}.
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
using U=uint32_t;
static U adj[31], bestmask; static int best;
int pc(U a){return __builtin_popcount(a);}
void bk(U P,U X,U Q){
 if(!P && !X){if(pc(Q)>best){best=pc(Q);bestmask=Q;}return;}
 if(pc(P)+pc(Q)<=best)return;
 U Uset=P|X; int pivot=-1,mx=-1;
 while(Uset){int v=__builtin_ctz(Uset);Uset&=Uset-1;int n=pc(P&adj[v]);if(n>mx){mx=n;pivot=v;}}
 U choices=P & ~(pivot<0?0:adj[pivot]);
 while(choices){int v=__builtin_ctz(choices);U b=U(1)<<v;choices-=b;
  bk(P&adj[v],X&adj[v],Q|b);
  P &=~b;X|=b;
  if(pc(P)+pc(Q)<=best)break;
 }
}
// 10 basis wedges in order 01,02,03,04,12,13,14,23,24,34.
int coeff(U mask,U x,U y){int ret=0;int bit=0;for(int i=0;i<5;i++)for(int j=i+1;j<5;j++,bit++)if(mask&(1U<<bit))ret ^= (((x>>i)&1)&((y>>j)&1)) ^ (((x>>j)&1)&((y>>i)&1));return ret;}
int main(){std::vector<int> cs;for(int c=0;c<1024;c++)if(c!=0&&c!=129&&c!=66&&c!=195 && c< (c^129) && c<(c^66) && c<(c^195))cs.push_back(c);
 if(cs.size()!=255){std::cerr<<"bad domain "<<cs.size()<<"\n";return 2;}
 int hist[32]={};std::cout<<"C,omega,clique_mask\n";
 for(int c:cs){for(int i=0;i<31;i++)adj[i]=0;for(int x=1;x<=31;x++)for(int y=x+1;y<=31;y++){if(coeff(129,x,y)||coeff(66,x,y)||coeff(c,x,y)){adj[x-1]|=U(1)<<(y-1);adj[y-1]|=U(1)<<(x-1);}}
 best=0;bestmask=0;bk(0x7fffffffU,0,0);hist[best]++;std::cout<<c<<","<<best<<","<<bestmask<<"\n";
 }
 std::cerr<<"domain=255 ";for(int k=0;k<32;k++)if(hist[k])std::cerr<<k<<":"<<hist[k]<<" ";std::cerr<<"\n";return 0;}
