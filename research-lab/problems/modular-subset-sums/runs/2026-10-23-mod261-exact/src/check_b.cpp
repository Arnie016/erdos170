#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
constexpr int MOD=261,MAX=130,SIZE=8;
using Mask=bitset<MOD>;using Row=array<int,SIZE>;
array<unsigned long long,SIZE+1> visits{};
vector<Row> leaves;
Mask shift(const Mask& d,int step){return (d<<step)|(d>>(MOD-step));}
int reduced(int x){x%=MOD;if(x<0)x+=MOD;return min(x,MOD-x);}
int direct_count(const Row& row){set<int> sums;for(int m=0;m<(1<<SIZE);m++){int s=0;for(int i=0;i<SIZE;i++)if(m&(1<<i))s+=row[i];sums.insert(s%MOD);}return (int)sums.size();}
Row transformed(const Row&a,int u){Row b;for(int i=0;i<SIZE;i++)b[i]=reduced(a[i]*u);sort(b.begin(),b.end());return b;}
void extend(int last,int count,const Mask& diffs,int threes,Row&a){
 visits[count]++;
 if(count==SIZE){if(direct_count(a)!=256)throw runtime_error("signed-difference leaf invalid");leaves.push_back(a);return;}
 for(int x=last+1;x<=MAX-(SIZE-count-1);x++){
  // Independent arithmetic eligibility, using 261=9*29.
  if(x%9==0 || x%29==0 || (x%3==0 && threes!=0))continue;
  if(diffs[x])continue;
  a[count]=x;
  extend(x,count+1,diffs|shift(diffs,x)|shift(diffs,MOD-x),threes+(x%3==0),a);
 }
}
void arrayjson(ostream&f,const Row&r){f<<'[';for(int i=0;i<SIZE;i++){if(i)f<<',';f<<r[i];}f<<']';}
int main(int argc,char**argv){try{
 if(argc!=2)throw runtime_error("expected results directory");string out=argv[1];auto start=chrono::steady_clock::now();
 for(int pos=0;pos<MOD;pos++){Mask z;z[pos]=1;for(int step=1;step<MOD;step++){auto y=shift(z,step);if(y.count()!=1||!y[(pos+step)%MOD])throw runtime_error("shift smoke");}}
 Row row{};row[0]=1;Mask d;d[0]=d[1]=d[MOD-1]=1;extend(1,1,d,0,row);
 vector<Row> prior;ifstream f(out+"/worker_a_sets.txt");if(!f)throw runtime_error("missing first census");
 while(true){Row a{};if(!(f>>a[0]))break;for(int i=1;i<SIZE;i++)if(!(f>>a[i]))throw runtime_error("truncated first census");prior.push_back(a);}
 if(prior!=leaves)throw runtime_error("FULL LIST DISAGREEMENT");
 if(leaves.empty())throw runtime_error("missing baseline");
 auto corrupt=prior;corrupt.pop_back();if(corrupt==leaves)throw runtime_error("negative list control failure");
 vector<int> units;for(int u=1;u<MOD;u++)if(gcd(u,MOD)==1)units.push_back(u);
 set<Row> full;for(const auto&a:leaves)for(int u:units)full.insert(transformed(a,u));
 Row dyadic{};for(int i=0;i<SIZE;i++)dyadic[i]=1<<i;
 set<Row> targets;for(int u:units)targets.insert(transformed(dyadic,u));
 map<Row,int> orbits;map<int,int> overlaps;map<int,int> nonunits;
 vector<Row> unmatched;
 const array<Row,3> templates={Row{1,2,4,8,16,32,64,128},Row{1,2,4,8,16,32,64,129},Row{1,2,4,8,16,32,65,129}};
 set<Row> template_members;for(const auto&a:templates){if(direct_count(a)!=256)throw runtime_error("proposed template invalid");for(int u:units)template_members.insert(transformed(a,u));}
 ofstream raw(out+"/full_sign_normalized_sets.txt");
 for(const auto&a:full){
  if(direct_count(a)!=256)throw runtime_error("unit expansion invalid");
  int num=0;for(int x:a)num+=(gcd(x,MOD)!=1);nonunits[num]++;
  Row key=a;for(int u:units)key=min(key,transformed(a,u));orbits[key]++;
  int best=0;for(const auto&t:targets){int c=0;for(int x:a)c+=binary_search(t.begin(),t.end(),x);best=max(best,c);}overlaps[best]++;
  if(!template_members.contains(a))unmatched.push_back(a);
  for(int i=0;i<SIZE;i++)raw<<a[i]<<(i+1==SIZE?'\n':' ');
 }
 Row bad{1,2,3,4,5,6,7,8};int bad_count=direct_count(bad);if(bad_count==256)throw runtime_error("negative collision control failure");
 ofstream j(out+"/worker_b.json");j<<"{\"completed\":true,\"N\":261,\"n\":8,\"full_lists_agree\":true,\"contains_one_count\":"<<leaves.size()<<",\"nodes\":[";
 for(int i=0;i<=SIZE;i++){if(i)j<<',';j<<visits[i];}j<<"],\"canonical_count\":"<<full.size()<<",\"signed_set_count\":"<<(full.size()*256)<<",\"orbit_classes\":[";
 bool first=true;for(const auto&[a,n]:orbits){if(!first)j<<',';first=false;j<<"{\"representative\":";arrayjson(j,a);j<<",\"canonical_size\":"<<n<<'}';}
 auto hist=[&](const map<int,int>&h){j<<'{';bool fi=true;for(auto[x,y]:h){if(!fi)j<<',';fi=false;j<<'"'<<x<<"\":"<<y;}j<<'}';};
 j<<"],\"overlap_histogram\":";hist(overlaps);j<<",\"nonunit_count_histogram\":";hist(nonunits);
 j<<",\"outside_published_templates\":[";for(size_t i=0;i<unmatched.size();i++){if(i)j<<',';arrayjson(j,unmatched[i]);}
 j<<"],\"negative_control_subset_sums\":"<<bad_count<<",\"removed_leaf_rejected\":true,\"seconds\":"<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<"}\n";
 if(!j||!raw)throw runtime_error("output error");
 cout<<"COMPLETE contains1="<<leaves.size()<<" canonical="<<full.size()<<" signed="<<full.size()*256<<" orbits="<<orbits.size()<<" unmatched="<<unmatched.size()<<"\n";
 }catch(const exception&e){cerr<<e.what()<<'\n';return 1;}}
