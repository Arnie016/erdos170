// Exact 2-replacement neighborhood, bitset approach. Input is frozen A below from candidate.json.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>
using namespace std;
using Mask=array<uint64_t,8>;
static constexpr int N=500;
static const vector<int> A={0,1,2,3,4,5,6,13,26,39,52,65,78,91,118,145,172,199,226,253,280,307,334,361,388,402,416,430,444,458,472,486,487,488,489,490,491,492,500};
inline void put(Mask &m, int d){if(d>0 && d<=N)m[d>>6]|=1ULL<<(d&63);}
inline int missing(const Mask &m){int n=0; for(int i=0;i<7;i++)n+=64-__builtin_popcountll(m[i]);n+=53-__builtin_popcountll(m[7] & ((1ULL<<53)-1));return n+1 /*bit zero was counted missing, remove afterwards*/-1;}
// Simpler explicit mask: bits 1..500 must be set.
inline int miss(const Mask &m){int ans=0;for(int d=1;d<=500;d++)ans+=( (m[d>>6]>>(d&63))&1ULL)==0;return ans;}
int main(){
 bool used[501]={}; for(int x:A){if(x<0||x>500||used[x])return 10;used[x]=true;}
 if(A.size()!=39||A.front()!=0||A.back()!=500)return 11;
 vector<int> U;for(int x=0;x<=500;x++)if(!used[x])U.push_back(x);
 const vector<int> S={7,493,494,495,496,497,498,499};for(int s:S)if(used[s])return 12;
 Mask baseline{};for(int i=0;i<39;i++)for(int j=0;j<i;j++)put(baseline,A[i]-A[j]);
 cout<<"baseline_missing="<<miss(baseline)<<" unused="<<U.size()<<" special="<<S.size()<<'\n';
 uint64_t checked=0,one_checked=0,complete=0;int min_missing=501; vector<int> witness;
 // Independent 1-replacement baseline over all interior removals.
 for(int i=1;i<38;i++){
  Mask c{};for(int j=0;j<39;j++)if(j!=i)for(int k=0;k<j;k++)if(k!=i)put(c,A[j]-A[k]);
  for(int t:U){Mask d=c;for(int j=0;j<39;j++)if(j!=i)put(d,abs(t-A[j]));one_checked++;if(miss(d)==0){cerr<<"unexpected one-swap";return 20;}}
 }
 for(int i=1;i<38;i++) for(int j=i+1;j<38;j++){
  vector<int> R;R.reserve(37);for(int k=0;k<39;k++)if(k!=i&&k!=j)R.push_back(A[k]);
  Mask base{};for(int k=0;k<37;k++)for(int h=0;h<k;h++)put(base,R[k]-R[h]);
  for(int s:S){
   Mask withS=base;for(int a:R)put(withS,abs(s-a));
   for(int t:U){if(t==s)continue; // count pairs S-S just once
    if(t>=493&&t<=499 || t==7){if(t<s)continue;}
    Mask m=withS;put(m,abs(t-s));for(int a:R)put(m,abs(t-a));
    ++checked;int nmiss=miss(m);min_missing=min(min_missing,nmiss);
    if(nmiss==0){++complete;if(witness.empty()){witness=R;witness.push_back(s);witness.push_back(t);sort(witness.begin(),witness.end());}}
   }
  }
 }
 cout<<"one_checked="<<one_checked<<" checked="<<checked<<" complete="<<complete<<" min_missing="<<min_missing<<"\n";
 if(!witness.empty()){cout<<"witness=";for(int a:witness)cout<<a<<",";cout<<"\n";}
 if(checked!=2437560ULL||one_checked!=17094ULL)return 30;
 return 0;
}
