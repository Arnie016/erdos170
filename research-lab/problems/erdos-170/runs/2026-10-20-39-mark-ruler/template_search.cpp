#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>
using namespace std;
// PREDECLARED CONTRACT: 39 marks exactly, positions 0 and 500,
// all distances 1..500. Exact template domain W(r,s) r=4..8,
// s=36-4r, one gap, two gaps with first delta in [-max(35,abs(D)),max(35,abs(D))].
// Distances checked by fresh pair loop independent of transformation.
const int N=500, M=39; long long tested=0; int best_missing=501;
vector<int> best_marks,winning_marks;int win_r=-1;string win_kind;
int eval(const vector<int>& gaps,int r,string kind){
    ++tested;int x=0;array<int,M> a{};a[0]=0;
    for(int i=0;i<M-1;++i){x+=gaps[i];a[i+1]=x;}
    if(x!=N)return 501;
    uint64_t D[8]={};
    for(int i=0;i<M;++i)for(int j=0;j<i;++j){int d=a[i]-a[j];if(d>0&&d<=500)D[d>>6]|=1ULL<<(d&63);}
    int have=0;
    for(int j=0;j<8;++j){uint64_t mask=(j==0?~1ULL:~0ULL); if(j==7)mask &= ((1ULL<<53)-1ULL); have+=__builtin_popcountll(D[j]&mask);}
    int missing=N-have;
    if(missing<best_missing){best_missing=missing;best_marks.assign(a.begin(),a.end());cerr<<"BEST "<<missing<<" r="<<r<<" kind="<<kind<<" trials="<<tested<<"\n";}
    if(missing==0){winning_marks.assign(a.begin(),a.end());win_r=r;win_kind=kind;return 0;}
    return missing;
}
vector<int> whichmann(int r,int s){
    vector<int> g;
    for(int i=0;i<r;++i)g.push_back(1);
    g.push_back(r+1);
    for(int i=0;i<r;++i)g.push_back(2*r+1);
    for(int i=0;i<s;++i)g.push_back(4*r+3);
    for(int i=0;i<r+1;++i)g.push_back(2*r+2);
    for(int i=0;i<r;++i)g.push_back(1);
    return g;
}
string jsonarr(const vector<int>& v){ostringstream o;o<<"[";for(size_t i=0;i<v.size();++i){if(i)o<<",";o<<v[i];}o<<"]";return o.str();}
int main(int argc,char**argv){
    auto t0=chrono::steady_clock::now();
    mt19937_64 rng(20261020);
    int tested_exact=0, tested_random=0;
    for(int r=4;r<=8;++r){
        int s=36-4*r;auto original=whichmann(r,s);if((int)original.size()!=38){cerr<<"ERROR wrong count\n";return 2;}
        int D=500-accumulate(original.begin(),original.end(),0);int H=max(35,abs(D));
        for(int i=0;i<38;++i){auto g=original;g[i]+=D;if(g[i]>0){if(eval(g,r,"one-gap")==0)goto finished;}}
        for(int i=0;i<38;++i)for(int j=i+1;j<38;++j){
            for(int u=-H;u<=H;++u){int v=D-u;if(original[i]+u<=0||original[j]+v<=0)continue;
                auto g=original;g[i]+=u;g[j]+=v;
                if(eval(g,r,"two-gap")==0)goto finished;
            }
        }
    }
    tested_exact=tested;
    // Predeclared 200,000 seeded random three-gap perturbations of r=7, s=8
    // around the Wichmann length-503 template, with u,v in [-10,10].
    {
      auto original=whichmann(7,8);int D=500-accumulate(original.begin(),original.end(),0);
      for(int it=0;it<200000;++it){
        int i=rng()%38,j=rng()%38,k=rng()%38;
        if(i==j||i==k||j==k)continue;
        int u=(int)(rng()%21)-10,v=(int)(rng()%21)-10,w=D-u-v;
        auto g=original;g[i]+=u;g[j]+=v;g[k]+=w;
        if(min({g[i],g[j],g[k]})<=0)continue;
        ++tested_random;
        if(eval(g,7,"three-gap-sample")==0)goto finished;
      }
    }
finished:
    double seconds=chrono::duration<double>(chrono::steady_clock::now()-t0).count();
    cout<<"{\"timestamp\":\"2026-10-20\",\"seed\":20261020,\"exact_evaluations\":"<<tested_exact
        <<",\"random_evaluations\":"<<tested_random<<",\"total_evaluations\":"<<tested
        <<",\"best_missing\":"<<best_missing<<",\"best_marks\":"<<jsonarr(best_marks)
        <<",\"found\":"<<(winning_marks.empty()?"false":"true")
        <<",\"winning_marks\":"<<jsonarr(winning_marks)<<",\"winning_r\":"<<win_r
        <<",\"winning_kind\":\""<<win_kind<<"\",\"wall_seconds\":"<<seconds<<"}"<<endl;
    return 0;
}
