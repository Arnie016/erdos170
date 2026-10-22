// Independent brute logical membership checker. Does not use bitsets or worker A logic.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>
using namespace std;
constexpr int L=500;
const array<int,39> pos={0,1,2,3,4,5,6,13,26,39,52,65,78,91,118,145,172,199,226,253,280,307,334,361,388,402,416,430,444,458,472,486,487,488,489,490,491,492,500};
int main(){
 array<int,501> inside{};for(int x:pos){if(x<0||x>500||inside[x])return 10;inside[x]=1;}
 if(!inside[0]||!inside[500])return 11;
 vector<int> out;for(int x=0;x<501;x++)if(!inside[x])out.push_back(x);
 vector<int> special;for(int d=0;d<=7;d++){int x=d==7?7:493+d;if(!inside[x])special.push_back(x);}sort(special.begin(),special.end());
 if(special.size()!=8||out.size()!=462)return 12;
 uint64_t total=0,success=0;int minimum=501;
 for(int a=1;a<38;a++)for(int b=a+1;b<38;b++){
  array<int,501> present=inside;present[pos[a]]=0;present[pos[b]]=0;
  array<int,501> cnt{};
  for(int u=0;u<501;u++)if(present[u])for(int v=u+1;v<501;v++)if(present[v])cnt[v-u]++;
  for(int s:special){
   array<int,501> cntS=cnt;
   for(int u=0;u<501;u++)if(present[u])cntS[abs(s-u)]++;
   present[s]=1;
   vector<int> miss;for(int k=1;k<=500;k++)if(cntS[k]==0)miss.push_back(k);
   for(int t:out){if(t==s || (t==7 || t>=493) && t<s)continue;
    ++total;int rem=0;
    for(int d:miss){
     bool covers=(t>=d && present[t-d]) || (t+d<=500 && present[t+d]);
     rem+=!covers;
    }
    minimum=min(minimum,rem);
    if(rem==0){++success;cout<<"FOUND remove="<<pos[a]<<","<<pos[b]<<" add="<<s<<","<<t<<"\n";}
   }
   present[s]=0;
  }
 }
 cout<<"checked="<<total<<" complete="<<success<<" min_missing="<<minimum<<"\n";
 return total==2437560ULL?0:30;
}
