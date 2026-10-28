#include "common.hpp"
#include <iostream>
#include <map>
using namespace std;
int main(){
 for(int q: {3,5}){
  Graph H=make_graph(q);long long count=0,minimum=999999,equality=0;map<int,int>hist;
  for(int ai=0;ai<(int)H.absolutes.size();ai++){
   int a=H.absolutes[ai];
   for(int bi=0;bi<(int)H.absolutes.size();bi++)for(int ci=bi+1;ci<(int)H.absolutes.size();ci++){
    int b=H.absolutes[bi],c=H.absolutes[ci];if(a==b || a==c)continue;
    for(auto [u,v]:H.edges){
     auto x=H.adj;
     x[u][v]=x[v][u]=0;
     x[a][b]=x[b][a]=x[a][c]=x[c][a]=1;
     long long nc=count_foursets(x);
     count++;minimum=min(minimum,nc);hist[(int)nc]++;if(nc==2*q-3)equality++;
     if(nc<2*q-3)throw runtime_error("counterexample");
    }
   }
  }
  long long expected=(long long)(q+1)*q*(q-1)/2*H.edges.size();
  if(count!=expected||minimum!=2*q-3||equality==0)throw runtime_error("domain/control mismatch");
  cout<<"q="<<q<<" graphs="<<count<<" min="<<minimum<<" equality="<<equality<<" histogram=";
  for(auto [k,v]:hist)cout<<k<<":"<<v<<",";cout<<"\n";
 }
}
