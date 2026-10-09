#include <bits/stdc++.h>
using namespace std; using U=uint32_t; using UU=uint64_t;
static U nei[32];
static U WEDGE[32][32];
struct Sub {U mask; vector<int> constraints;};
vector<Sub> subs;
int best;U witness;
inline int bc(U a){return __builtin_popcount(a);}
void bk(U P,U X,U R,int r){
 if(!P&&!X){if(r>best){best=r;witness=R;}return;}
 if(r+bc(P)<=best)return;
 U both=P|X;int pivot=0,score=-1;while(both){int u=__builtin_ctz(both);both&=both-1;int q=bc(P&nei[u]);if(q>score){score=q;pivot=u;}}
 U todo=P&~nei[pivot];
 while(todo){int u=__builtin_ctz(todo);U bit=U(1)<<u;todo&=todo-1;
  bk(P&nei[u],X&nei[u],R|bit,r+1);P&=~bit;X|=bit;
  if(r+bc(P)<=best)return;
 }
}
static vector<U> covers;
static vector<int> byv[32];
static unordered_set<UU> memo;
static int recursive_nodes;
bool set_cover(U uncovered,int depth){
 if(!uncovered)return true;
 if(depth<=0)return false;
 UU key=(UU(uncovered)<<6)|unsigned(depth);
 if(memo.find(key)!=memo.end())return false;
 int maxgain=0;for(U m:covers)maxgain=max(maxgain,bc(m&uncovered));
 if(maxgain*depth<bc(uncovered)){memo.insert(key);return false;}
 int pivot=-1,least=INT_MAX;
 U left=uncovered;
 while(left){int v=__builtin_ctz(left);left&=left-1;int q=byv[v].size();if(q<least){least=q;pivot=v;}}
 if(pivot<0||least==0){memo.insert(key);return false;}
 vector<pair<int,int>> options;options.reserve(least);
 for(int id:byv[pivot]){int gain=bc(covers[id]&uncovered); if(gain)options.emplace_back(-gain,id);}
 sort(options.begin(),options.end());
 for(auto [g,id]:options){if(set_cover(uncovered&~covers[id],depth-1))return true;}
 memo.insert(key);return false;
}
void gen_subspaces(){
 unordered_set<U> seen;queue<U> q; q.push(1);seen.insert(1);
 while(!q.empty()){
  U m=q.front();q.pop();
  vector<int> basis;U span=1;
  for(int v=1;v<32;v++)if((m>>v&1)&&!(span>>v&1)){
   U newspan=span;U temp=span;while(temp){int x=__builtin_ctz(temp);temp&=temp-1;newspan|=U(1)<<(x^v);} span=newspan;basis.push_back(v);
  }
  if(span!=m){cerr<<"BAD_SUBSPACE\n";exit(7);}
  vector<int> cs;for(int i=0;i<(int)basis.size();i++)for(int j=i+1;j<(int)basis.size();j++)cs.push_back(WEDGE[basis[i]][basis[j]]);
  subs.push_back({m&~U(1),cs});
  for(int v=1;v<32;v++)if(!(m>>v&1)){
   U t=m;U temp=m;while(temp){int x=__builtin_ctz(temp);temp&=temp-1;t|=U(1)<<(x^v);}
   if(seen.insert(t).second)q.push(t);
  }
 }
 if(subs.size()!=374){cerr<<"BAD_SUBSPACE_COUNT "<<subs.size()<<"\n";exit(8);}
}
UU canonkey(int a,int b){int c=a^b;array<int,3> ar={a,b,c};sort(ar.begin(),ar.end());return (UU(ar[0])<<20)|(UU(ar[1])<<10)|ar[2];}
int main(){
 int idx=0;for(int i=0;i<5;i++)for(int j=i+1;j<5;j++){for(int x=0;x<32;x++)for(int y=0;y<32;y++){if(((x>>i&1)&(y>>j&1))^((x>>j&1)&(y>>i&1)))WEDGE[x][y]|=1<<idx;}idx++;}
 gen_subspaces();
 ifstream in("results/rank5_pencils.jsonl");if(!in){cerr<<"MISSING_GENERATOR_RECEIPT\n";return 10;}
 unordered_map<UU,int> expected;expected.reserve(200000);string s;int ca,cb,om;
 while(getline(in,s)){
  bool colorable=(s.find("\"colorable\":true")!=string::npos);
  if(sscanf(s.c_str(),"{\"a\":%d,\"b\":%d,\"omega\":%d",&ca,&cb,&om)!=3||!colorable){cerr<<"BAD_RECORD\n";return 11;}
  if(!expected.emplace(canonkey(ca,cb),om).second){cerr<<"DUPLICATE_RECEIPT\n";return 12;}
 }
 if(expected.size()!=174251){cerr<<"INCOMPLETE_RECEIPT "<<expected.size()<<"\n";return 13;}
 int hist[32]={};int checked=0, no_covers=0, maxNodes=0;
 for(int a=1;a<1024;a++)for(int b=a+1;b<1024;b++){
  int c=a^b;if(c<=b)continue;
  UU key=canonkey(a,b); auto it=expected.find(key);if(it==expected.end()){cerr<<"MISSING_KEY\n";return 14;}
  for(int x=1;x<32;x++){U m=0;for(int y=1;y<32;y++)if(x!=y&&((bc(a&WEDGE[x][y])&1)||(bc(b&WEDGE[x][y])&1)))m|=U(1)<<y;nei[x]=m;}
  best=0;witness=0;bk(0xfffffffeu,0,0,0);if(best!=it->second){cerr<<"OMEGA_DISAGREEMENT "<<a<<" "<<b<<" "<<best<<" "<<it->second<<"\n";return 15;}
  vector<U> all;
  for(auto &sub:subs){bool iso=true;for(int w:sub.constraints)if((bc(a&w)&1)||(bc(b&w)&1)){iso=false;break;}if(iso)all.push_back(sub.mask);}
  covers.clear();for(U m:all){bool dominated=false;for(U n:all)if(n!=m && (m&n)==m){dominated=true;break;}if(!dominated)covers.push_back(m);}
  for(int v=1;v<32;v++)byv[v].clear();for(int i=0;i<(int)covers.size();i++){U m=covers[i];while(m){int v=__builtin_ctz(m);m&=m-1;byv[v].push_back(i);}}
  memo.clear();recursive_nodes=0;
  if(!set_cover(0xfffffffeu,best)){cerr<<"COVER_GAP "<<a<<" "<<b<<" "<<best<<"\n";no_covers++;return 16;}
  checked++;hist[best]++;if(checked%5000==0)cerr<<"certified "<<checked<<"\n";
 }
 cout<<"{\"status\":\"INDEPENDENT_PASS\",\"checked\":"<<checked<<",\"gap_count\":"<<no_covers<<",\"subspaces\":"<<subs.size()<<",\"histogram\":{";
 for(int j=0;j<32;j++)if(hist[j])cout<<"\""<<j<<"\":"<<hist[j]<<",";cout<<"\"end\":0}}"<<endl;
 return checked==174251?0:17;
}
