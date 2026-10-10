// Independent checker v1, locally compiled and run, C++17
// Direct coordinate bilinear calculation, Bron-Kerbosch exact clique bound,
// plus raw clique/cover-witness validation for one faithful rank-three pencil.
#include <bits/stdc++.h>
using namespace std;
using U=uint32_t;
int forms[3], adj[32][32],omega=0;U N[31];int countBK=0;
int B(int x,int y){int code=0,k=0;for(int i=0;i<5;i++)for(int j=i+1;j<5;j++,k++)if(((x>>i&1)&&(y>>j&1))^((x>>j&1)&&(y>>i&1))){for(int t=0;t<3;t++)code^=((forms[t]>>k)&1)<<t;}return code;}
void bk(U R,U P,U X){if(!P&&!X){omega=max(omega,__builtin_popcount(R));countBK++;return;}U Q=P|X;int pivot=-1,md=-1;while(Q){int v=__builtin_ctz(Q);Q&=Q-1;int d=__builtin_popcount(P&N[v]);if(d>md){md=d;pivot=v;}}U candidates=P&(pivot<0?U(-1):~N[pivot]);while(candidates){int v=__builtin_ctz(candidates);U vbit=U(1)<<v;candidates&=~vbit;bk(R|vbit,P&N[v],X&N[v]);P&=~vbit;X|=vbit;}}
int main(int argc,char**argv){if(argc!=4){cerr<<"usage: a b c; checker reads first JSONL record via regex from results/search.jsonl\n";return 2;}for(int j=0;j<3;j++)forms[j]=stoi(argv[j+1]);if(!(forms[0]&&forms[1]&&forms[2])||forms[0]==forms[1]||forms[0]==forms[2]||forms[1]==forms[2]||forms[2]==(forms[0]^forms[1]))return 3;
ifstream in("results/search.jsonl");string line;bool found=false;int claimOmega,claimChi;vector<int> clique;vector<vector<int>> cover;regex ra("\\\"a\\\":([0-9]+)"),rb("\\\"b\\\":([0-9]+)"),rc("\\\"c\\\":([0-9]+)"),rw("\\\"omega\\\":([0-9]+)"),rh("\\\"chi\\\":([0-9]+)");smatch m;
while(getline(in,line)){auto capture=[&](regex &r){smatch m;if(!regex_search(line,m,r))throw runtime_error("field missing");return stoi(m[1]);};if(capture(ra)!=forms[0]||capture(rb)!=forms[1]||capture(rc)!=forms[2])continue;claimOmega=capture(rw);claimChi=capture(rh);size_t cs=line.find("\"clique\":["),col=line.find("\"classes\":[");if(cs==string::npos||col==string::npos)return 4;string cli=line.substr(cs+10,line.find("]",cs)-(cs+10));for(char&c:cli)if(c==',')c=' ';stringstream ss(cli);int x;while(ss>>x)clique.push_back(x);size_t pos=col+11;while(pos<line.size()&&line[pos]!=']'){while(pos<line.size()&&line[pos]!= '['){if(line[pos]==']')break;pos++;}if(pos>=line.size()||line[pos]!= '[')break;size_t end=line.find(']',pos);if(end==string::npos)return 4;string s=line.substr(pos+1,end-pos-1);for(char&c:s)if(c==',')c=' ';stringstream is(s);vector<int> v;while(is>>x)v.push_back(x);cover.push_back(v);pos=end+1;if(line[pos]==',')pos++;}found=true;break;}
if(!found){cerr<<"missing_exact_record\n";return 5;}
for(int x=1;x<32;x++){if(B(x,x))return 6;for(int y=1;y<32;y++)if(B(x,y)!=B(y,x))return 7;}
for(int x=1;x<32;x++){bool rad=true;for(int y=1;y<32;y++)if(B(x,y)){rad=false;break;}if(rad){cerr<<"nonfaithful\n";return 8;}}
for(int x=1;x<32;x++)for(int y=x+1;y<32;y++)if(!B(x,y))for(int z=y+1;z<32;z++)if(z!=(x^y)&&!B(x,z)&&!B(y,z)){cerr<<"isotropic_3space\n";return 9;}
for(int x=1;x<32;x++)for(int y=1;y<32;y++)adj[x][y]=B(x,y)!=0;
for(int x=1;x<32;x++){N[x-1]=0;for(int y=1;y<32;y++)if(adj[x][y])N[x-1]|=U(1)<<(y-1);}bk(0,(U(1)<<31)-1,0);
if(omega!=claimOmega){cerr<<"clique_mismatch_"<<omega<<"_"<<claimOmega<<"\n";return 10;}
if((int)clique.size()!=omega)return 11;for(int i=0;i<(int)clique.size();i++)for(int j=i+1;j<(int)clique.size();j++)if(!adj[clique[i]][clique[j]])return 12;
if((int)cover.size()!=claimChi)return 13;U seen=0;for(auto&cls:cover){if(cls.empty())return 14;for(int x:cls){if(x<1||x>31||seen>>(x-1)&1)return 15;seen|=U(1)<<(x-1);for(int y:cls)if(x!=y&&adj[x][y])return 16;}}if(seen!=((U(1)<<31)-1))return 17;
if(claimChi!=omega){cerr<<"VALID_UPPER_AND_CLIQUE_BUT_CHROMATIC_MINIMALITY_UNCHECKED\n";return 18;}
cout<<"INDEPENDENT_CERTIFICATE_PASS omega=chi="<<omega<<" maximal_cliques="<<countBK<<" 31_vertices=yes nonzero_radical=0 no_isotropic_3space=yes\n";return 0;
}
