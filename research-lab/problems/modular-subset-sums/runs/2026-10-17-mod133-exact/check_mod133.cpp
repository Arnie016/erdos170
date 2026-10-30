#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>
using namespace std;
constexpr int N=133, H=66, K=7;
using Bits=bitset<N>;
array<long long,K+1> nodes{};
vector<array<int,K>> accepted;
map<int,int> overlap_hist;
map<vector<int>,int> orbit_hist;
map<vector<int>,array<int,K>> orbit_witness;
Bits rol(const Bits &b, int k){ return (b<<k)|(b>>(N-k)); }
int norm(int x){x%=N;if(x<0)x+=N;return min(x,N-x);}
vector<int> transform(const array<int,K> &a,int u){vector<int>b;for(int v:a)b.push_back(norm(u*v));sort(b.begin(),b.end());return b;}
void leaf(const array<int,K>& a){
    set<int> sums{0};
    for(int v:a){vector<int> extra;for(int t:sums) extra.push_back((t+v)%N); for(int t:extra) sums.insert(t);}
    if(sums.size()!=128){cerr<<"DIFFERENCE MASK CHECK ERROR at leaf\n";exit(10);}
    int mx=0; vector<int> bestOrbit;
    for(int u=1;u<N;++u){if(gcd(u,N)!=1)continue;
        set<int> orbit;for(int j=0;j<K;++j)orbit.insert(norm(u*(1<<j)));
        if(orbit.size()!=K){cerr<<"degenerate dyadic orbit\n";exit(11);}
        int overlap=0;for(int v:a) if(orbit.contains(v))overlap++;
        mx=max(mx,overlap);
        auto equiv=transform(a,u);
        if(bestOrbit.empty() || equiv<bestOrbit) bestOrbit=equiv;
    }
    overlap_hist[mx]++;
    orbit_hist[bestOrbit]++;
    orbit_witness[bestOrbit]=a;
    accepted.push_back(a);
}
void dfs(int last, int depth, Bits D,array<int,K> &A){
    nodes[depth]++;
    if(depth==K){leaf(A);return;}
    for(int a=last+1;a<=H-(K-depth-1);++a){
        if(D.test(a))continue;
        Bits next=D|rol(D,a)|rol(D,N-a);
        A[depth]=a;
        dfs(a,depth+1,next,A);
    }
}
int main(int argc,char **argv){
 auto t0=chrono::steady_clock::now();
 if(argc!=2){cerr<<"need output path\n";return 2;}
 Bits D; D.set(0); array<int,K> A{};
 dfs(0,0,D,A);
 array<long long,K+1> expected={1,60,1830,36656,490780,3322318,2942381,162};
 if(nodes!=expected){cerr<<"COUNTS MISMATCH\n";for(auto i:nodes)cerr<<i<<" ";cerr<<"\n";return 3;}
 if(accepted.size()!=162 || overlap_hist.begin()->first < 6){cerr<<"BAD candidate classification\n";return 4;}
 if(orbit_hist.size()!=3){cerr<<"Unexpected orbit count "<<orbit_hist.size()<<"\n";return 5;}
 set<int> bad{0}; array<int,K> badA={1,2,3,4,5,6,7};
 for(int x:badA){vector<int> shifted;for(int s:bad)shifted.push_back((s+x)%N);for(int s:shifted)bad.insert(s);}
 if(bad.size()==128){cerr<<"negative control failure\n";return 6;}
 ofstream f(argv[1]);f<<"{\n\"N\":133,\"n\":7,\"completed\":true,\"nodes\":[";
 for(int i=0;i<=K;i++){if(i)f<<",";f<<nodes[i];}f<<"],\n\"overlap_hist\":{";
 bool first=true;for(auto [x,y]:overlap_hist){if(!first)f<<",";first=false;f<<"\""<<x<<"\":"<<y;}f<<"},\n\"orbit_classes\":[\n";
 first=true;for(auto &[key,count]:orbit_hist){if(!first)f<<",\n";first=false;f<<"{\"representative\":[";for(int i=0;i<K;++i){if(i)f<<",";f<<key[i];}f<<"],\"count\":"<<count<<"}";}f<<"],\n\"accepted\":[\n";
 first=true;for(auto a:accepted){if(!first)f<<",\n";first=false;f<<"[";for(int i=0;i<K;++i){if(i)f<<",";f<<a[i];}f<<"]";}
 f<<"],\n\"negative_control_distinct_subset_sums\":"<<bad.size()<<",\n\"wall_seconds\":"<<chrono::duration<double>(chrono::steady_clock::now()-t0).count()<<"\n}\n";
 f.close();
 cout<<"COMPLETE count="<<accepted.size()<<" orbits="<<orbit_hist.size()<<" nodes=";for(auto x:nodes)cout<<x<<",";
 cout<<" hist=";for(auto [k,v]:overlap_hist)cout<<k<<":"<<v<<",";cout<<" negatives="<<bad.size()<<"\n";
 for(auto &[rep,n]:orbit_hist){cout<<"orbit "<<n<<" : ";for(int x:rep)cout<<x<<" ";cout<<"\n";}
}
