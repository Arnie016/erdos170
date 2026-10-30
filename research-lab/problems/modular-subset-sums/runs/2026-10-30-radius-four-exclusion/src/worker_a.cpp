#include <bits/stdc++.h>
using namespace std;
constexpr int N=517;
using Bits=bitset<N>;
int invmod(int a){int t=0,nt=1,r=N,nr=a;while(nr){int q=r/nr;tie(t,nt)=make_tuple(nt,t-q*nt);tie(r,nr)=make_tuple(nr,r-q*nr);}if(r!=1)return -1;return (t%N+N)%N;}
struct Record{int a;Bits b;};
struct Result {vector<int>vertices; vector<pair<int,int>>edges;vector<int>max_clique;};
vector<int> holes(int r){vector<int> h={0,1,N-1,r,N-r};sort(h.begin(),h.end());return h;}
Result compute(int radius){auto H=holes(radius); Bits S;S.set();for(int x:H)S.reset(x);
 vector<Record> R;
 for(int a=1;a<N;a++)if(gcd(a,N)==1){int inv=invmod(a);vector<int> p;for(int h:H)p.push_back((h*inv)%N);sort(p.begin(),p.end());
   Bits B;bool ok=true;for(int i=0;i<5;i++){int d=(p[(i+1)%5]-p[i]+N)%N;if(!(d&1)){ok=false;break;}for(int k=1;k<d;k+=2){int x= ((p[i]+k)%N)*a%N;B.set(x);}}
   if(!ok)continue; if(B.count()!=256)throw runtime_error("half-set size");
   Bits U=B;for(int x=0;x<N;x++)if(B.test(x)){int y=(x+a)%N;if(U.test(y))throw runtime_error("duplicate a shift");U.set(y);}
   if(U!=S)throw runtime_error("bad one-shift split");R.push_back({a,B});
 }
 int n=R.size();vector<Bits> adj(n);Result out;
 for(const auto&x:R)out.vertices.push_back(x.a);
 for(int i=0;i<n;i++)for(int j=i+1;j<n;j++){
   Bits C=R[i].b & R[j].b;if(C.count()!=128)continue;
   Bits U;bool ok=true;int a=R[i].a,b=R[j].a;
   for(int x=0;x<N&&ok;x++)if(C.test(x)){
     int y[4]={x,(x+a)%N,(x+b)%N,(x+a+b)%N};
     for(int z:y){if(U.test(z)){ok=false;break;}U.set(z);}
   }
   if(!ok||U!=S)continue;
   adj[i].set(j);adj[j].set(i);out.edges.push_back({a,b});
 }
 vector<int>Q,P(n);iota(P.begin(),P.end(),0);
 function<void(vector<int>)> dfs=[&](vector<int>p){if(Q.size()>out.max_clique.size()){out.max_clique.clear();for(int x:Q)out.max_clique.push_back(R[x].a);}while(!p.empty()){
   if(Q.size()+p.size()<=out.max_clique.size())return;
   int v=p.back();p.pop_back();vector<int>next;for(int x:p)if(adj[v].test(x))next.push_back(x);
   Q.push_back(v);dfs(next);Q.pop_back();
 }};
 dfs(P);sort(out.max_clique.begin(),out.max_clique.end());return out;
}
void emit_result(ostream&o,const Result&r){o<<"{\"vertices\":[";for(int i=0;i<(int)r.vertices.size();i++){if(i)o<<',';o<<r.vertices[i];}o<<"],\"edges\":[";for(int i=0;i<(int)r.edges.size();i++){if(i)o<<',';o<<'['<<r.edges[i].first<<','<<r.edges[i].second<<']';}o<<"],\"clique_size\":"<<r.max_clique.size()<<",\"clique\":[";for(int i=0;i<(int)r.max_clique.size();i++){if(i)o<<',';o<<r.max_clique[i];}o<<"]}";}
int main(int argc,char**argv){if(argc!=2)return 2;auto t0=chrono::steady_clock::now();auto ctrl=compute(3);if(ctrl.vertices.size()!=156||ctrl.edges.size()!=308||ctrl.max_clique.size()!=9){cerr<<"CONTROL MISMATCH eligible "<<ctrl.vertices.size()<<" edges "<<ctrl.edges.size()<<" clique "<<ctrl.max_clique.size()<<'\n';return 3;}
 auto x=compute(4);ofstream f(argv[1]);if(!f)return 4;f<<"{\"control\":{";f<<"\"eligible\":"<<ctrl.vertices.size()<<",\"edges\":"<<ctrl.edges.size()<<",\"max_clique\":"<<ctrl.max_clique.size()<<"},\"target\":";emit_result(f,x);f<<",\"elapsed_seconds\":"<<chrono::duration<double>(chrono::steady_clock::now()-t0).count()<<"}\n";if(!f)return 5;cerr<<"A eligible="<<x.vertices.size()<<" edges="<<x.edges.size()<<" clique="<<x.max_clique.size()<<'\n';return 0;}
