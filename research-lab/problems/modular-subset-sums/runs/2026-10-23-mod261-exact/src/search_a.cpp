#include <array>
#include <bitset>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
constexpr int N=261,H=130,K=8;
using Bits=bitset<N>; using Row=array<int,K>;
array<unsigned long long,K+1> nodes{};
vector<Row> accepted;
Bits rot(const Bits& b,int k){return (b<<k)|(b>>(N-k));}
void smoke(){
 for(int i=0;i<N;i++){Bits b; b.set(i);for(int k=1;k<N;k++){auto t=rot(b,k); if(t.count()!=1||!t.test((i+k)%N))throw runtime_error("rotation smoke failure");}}
 Bits s;s.set(0);for(int i=0;i<K;i++){auto t=rot(s,1<<i);if((s&t).any())throw runtime_error("baseline collision");s|=t;}if(s.count()!=256)throw runtime_error("baseline count");
}
void dfs(int last,int depth,const Bits& s,int mult3,Row& a){
 nodes[depth]++;
 if(depth==K){accepted.push_back(a);return;}
 for(int x=last+1;x<=H-(K-depth-1);x++){
  if(gcd(x,N)>5 || (x%3==0 && mult3))continue;
  Bits shifted=rot(s,x);
  if((s&shifted).any())continue;
  a[depth]=x; dfs(x,depth+1,s|shifted,mult3+(x%3==0),a);
 }
}
int main(int argc,char** argv){
 try{
 if(argc!=2)throw runtime_error("expected results directory");
 const string out=argv[1];auto start=chrono::steady_clock::now();smoke();
 Row a{};a[0]=1;Bits s;s.set(0);s.set(1);dfs(1,1,s,0,a);
 ofstream f(out+"/worker_a_sets.txt");for(const auto& r:accepted){for(int i=0;i<K;i++)f<<r[i]<<(i+1==K?'\n':' ');}if(!f)throw runtime_error("write sets failure");f.close();
 ofstream j(out+"/worker_a.json");j<<"{\"completed\":true,\"N\":261,\"n\":8,\"contains_one_count\":"<<accepted.size()<<",\"nodes\":[";
 for(int i=0;i<=K;i++){if(i)j<<',';j<<nodes[i];}
 j<<"],\"seconds\":"<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<"}\n";
 if(!j)throw runtime_error("write JSON failure");cout<<"COMPLETE normalized leaves="<<accepted.size()<<"\n";
 }catch(const exception&e){cerr<<e.what()<<'\n';return 1;}
}
