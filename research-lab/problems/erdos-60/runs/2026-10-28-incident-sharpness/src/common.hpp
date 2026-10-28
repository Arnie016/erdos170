#ifndef E60_COMMON_HPP
#define E60_COMMON_HPP
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
using namespace std;
struct Point {int x,y,z;};
struct Graph {int q,n;vector<Point> p;vector<vector<unsigned char>> adj;vector<int> absolutes;vector<pair<int,int>> edges;};
Graph make_graph(int q){
 Graph H;H.q=q;
 for(int x=0;x<q;x++) for(int y=0;y<q;y++) H.p.push_back({1,x,y});
 for(int x=0;x<q;x++) H.p.push_back({0,1,x});
 H.p.push_back({0,0,1}); H.n=(int)H.p.size();
 if(H.n!=q*q+q+1)throw runtime_error("wrong projective points");
 H.adj.assign(H.n,vector<unsigned char>(H.n,0));
 auto dot=[&](int i,int j){auto a=H.p[i],b=H.p[j];int r=(a.x*b.z+a.z*b.x-2*a.y*b.y)%q;return (r+q)%q;};
 for(int i=0;i<H.n;i++){
  if(dot(i,i)==0)H.absolutes.push_back(i);
  for(int j=i+1;j<H.n;j++) if(dot(i,j)==0){H.adj[i][j]=H.adj[j][i]=1;H.edges.push_back({i,j});}
 }
 if((int)H.absolutes.size()!=q+1 || (int)H.edges.size()!=q*(q+1)*(q+1)/2)throw runtime_error("wrong graph counts");
 for(int i:H.absolutes)for(int j:H.absolutes)if(i!=j && H.adj[i][j])throw runtime_error("absolute adjacency");
 return H;
}
long long count_codegrees(const vector<vector<unsigned char>>& a){
 int n=(int)a.size();long long sum=0;
 for(int u=0;u<n;u++)for(int v=u+1;v<n;v++){
  int c=0;for(int w=0;w<n;w++)if(a[u][w] && a[v][w])c++;
  sum+=(long long)c*(c-1)/2;
 }
 if(sum%2)throw runtime_error("parity error");
 return sum/2;
}
long long count_foursets(const vector<vector<unsigned char>>& a){
 int n=(int)a.size();long long z=0;
 for(int i=0;i<n;i++)for(int j=i+1;j<n;j++)for(int k=j+1;k<n;k++)for(int l=k+1;l<n;l++){
  z+=(a[i][j] && a[j][k] && a[k][l] && a[l][i]);
  z+=(a[i][j] && a[j][l] && a[l][k] && a[k][i]);
  z+=(a[i][k] && a[k][j] && a[j][l] && a[l][i]);
 }
 return z;
}
#endif
