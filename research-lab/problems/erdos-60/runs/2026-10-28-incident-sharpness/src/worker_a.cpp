#include "common.hpp"
#include <iostream>
#include <map>
using namespace std;
int main(){
 for(int q: {3,5}){
  Graph H=make_graph(q); long long count=0,mincount=1LL<<50,equality=0;map<int,int>hist;
  for(int a:H.absolutes) for(int bi=0;bi<(int)H.absolutes.size();bi++){
   int b=H.absolutes[bi];if(a==b)continue;
   for(int ci=bi+1;ci<(int)H.absolutes.size();ci++){
    int c=H.absolutes[ci];if(a==c)continue;
    auto x=H.adj;x[a][b]=x[b][a]=1;x[a][c]=x[c][a]=1;
    if(count_codegrees(x)!=2*q-1)throw runtime_error("pre-deletion control");
    for(auto [u,v]:H.edges){
     x[u][v]=x[v][u]=0;
     auto nc=count_codegrees(x);
     x[u][v]=x[v][u]=1;
     if(nc<2*q-3)throw runtime_error("found counterexample");
     if(nc==2*q-3)equality++;
     mincount=min(mincount,nc);hist[(int)nc]++;count++;
    }
   }
  }
  long long expected=(long long)(q+1)*q*(q-1)/2*H.edges.size();
  if(count!=expected || equality==0 || mincount!=2*q-3)throw runtime_error("domain/control mismatch");
  cout<<"q="<<q<<" graphs="<<count<<" min="<<mincount<<" equality="<<equality<<" histogram=";
  for(auto [k,v]:hist)cout<<k<<":"<<v<<",";cout<<"\n";
 }
}
