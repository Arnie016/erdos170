// Independent reconstruction in matrix coordinates and isotropic-subspace cover search.
// Verifies raw clique masks from worker 1; writes constructive upper-bound certificates.
#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using U=uint32_t;
int pop(U x){return __builtin_popcount(x);}
static std::vector<U> spaces;
static std::array<std::array<uint8_t,5>,5> mats[3];
static std::vector<U> candidates;
static std::vector<int> at[31], result;
static std::unordered_map<U,int> seen;
static U ALL=0x7FFFFFFF;
void subspaces(){
 std::unordered_set<U> have; spaces.push_back(1);have.insert(1);
 for(size_t i=0;i<spaces.size();i++){
  U a=spaces[i];for(int v=1;v<32;v++)if(!(a&(U(1)<<v))){
   U b=a;for(int x=0;x<32;x++)if(a&(U(1)<<x))b|=(U(1)<<(x^v));
   if(have.insert(b).second)spaces.push_back(b);
  }
 }
 if(spaces.size()!=374){std::cerr<<"BAD_SPACES="<<spaces.size()<<"\n";std::exit(4);}
}
std::array<std::array<uint8_t,5>,5> matrix(int mask){
 std::array<std::array<uint8_t,5>,5> M{};
 int at=0;for(int i=0;i<5;i++)for(int j=i+1;j<5;j++,at++) if((mask>>at)&1){M[i][j]=1;M[j][i]=1;}
 return M;
}
int bil(int f,int x,int y){int v=0;for(int i=0;i<5;i++)if((x>>i)&1)for(int j=0;j<5;j++)if((y>>j)&1)v ^= mats[f][i][j];return v;}
bool can_cover(U covered,int rem){
 if(covered==ALL)return true;
 if(rem==0)return false;
 auto it=seen.find(covered);if(it!=seen.end() && it->second>=rem)return false;
 // Select the unvisited point with fewest available candidate gains, tie-break by maximal gain.
 int pick=-1,lowest=100000,topgain=-1;
 for(int i=0;i<31;i++)if(!(covered&(U(1)<<i))){
  int choices=0, gain=0;
  for(int j:at[i]){int n=pop(candidates[j]&~covered); if(n>0){choices++;gain=std::max(gain,n);}}
  if(choices<lowest || (choices==lowest&&gain>topgain)){pick=i;lowest=choices;topgain=gain;}
 }
 if(pick==-1)return true;
 std::vector<std::pair<int,int>> order;
 for(int j:at[pick]){int gain=pop(candidates[j]&~covered);if(gain)order.emplace_back(-gain,j);}
 std::sort(order.begin(),order.end());
 for(auto [neg,j]:order){result.push_back(j);if(can_cover(covered|candidates[j],rem-1))return true;result.pop_back();}
 seen[covered]=rem;return false;
}
int main(int argc,char**argv){if(argc!=3){std::cerr<<"usage: exe cliques.csv covers.csv\n";return 2;}
 subspaces();std::ifstream in(argv[1]);std::ofstream out(argv[2]);std::string line;std::getline(in,line);
 out<<"C,omega,cover_count,subspace_masks\n";
 int hist[32]={},done=0;
 while(std::getline(in,line)){
  std::stringstream ss(line);std::string a,b,d;std::getline(ss,a,',');std::getline(ss,b,',');std::getline(ss,d,',');
  int c=std::stoi(a),k=std::stoi(b);U mask=U(std::stoul(d));
  mats[0]=matrix(129);mats[1]=matrix(66);mats[2]=matrix(c);
  if(pop(mask)!=k){std::cerr<<"BAD_CLIQUE_CARD="<<c<<"\n";return 5;}
  for(int x=1;x<=31;x++)if(mask&(U(1)<<(x-1)))for(int y=x+1;y<=31;y++)if(mask&(U(1)<<(y-1))){
   if(!bil(0,x,y)&&!bil(1,x,y)&&!bil(2,x,y)){std::cerr<<"INVALID_CLIQUE="<<c<<"\n";return 6;}
  }
  // Build ALL isotropic linear subspaces of F_2^5 from 374 independently enumerated spaces.
  std::vector<U> iso;for(U S:spaces){bool ok=true;for(int x=1;x<=31 && ok;x++)if(S&(U(1)<<x))for(int y=x+1;y<=31;y++)if(S&(U(1)<<y))if(bil(0,x,y)||bil(1,x,y)||bil(2,x,y)){ok=false;break;}
   if(ok)iso.push_back(S>>1);
  }
  candidates.clear();for(U x:iso){bool maximal=true;for(U y:iso)if(x!=y && (x&y)==x){maximal=false;break;}if(maximal)candidates.push_back(x);}
  for(auto &v:at)v.clear();for(int j=0;j<(int)candidates.size();j++)for(int i=0;i<31;i++)if(candidates[j]&(U(1)<<i))at[i].push_back(j);
  for(int i=0;i<31;i++)if(at[i].empty()){std::cerr<<"UNCOVERABLE="<<c<<"\n";return 7;}
  result.clear();seen.clear();if(!can_cover(0,k)){std::cerr<<"NO_MATCHING_COVER="<<c<<" tested="<<done<<"\n";return 8;}
  if(result.size()>unsigned(k)){std::cerr<<"BAD_COVER_LENGTH"<<c<<"\n";return 9;}
  U union_mask=0;for(int j:result)union_mask|=candidates[j];if(union_mask!=ALL){std::cerr<<"BAD_COVER_UNION"<<c<<"\n";return 10;}
  out<<c<<","<<k<<","<<result.size()<<",";for(int j=0;j<(int)result.size();j++){if(j)out<<":";out<<candidates[result[j]];}out<<"\n";
  hist[k]++;done++;
  if(done==1)std::cerr<<"smoke: first pencil="<<c<<" confirmed, "<<spaces.size()<<" subspaces total\n";
 }
 if(done!=255){std::cerr<<"INCOMPLETE="<<done<<"\n";return 11;}
 std::cerr<<"completed="<<done<<" ";for(int i=0;i<32;i++)if(hist[i])std::cerr<<i<<":"<<hist[i]<<" ";std::cerr<<"\n";
}
