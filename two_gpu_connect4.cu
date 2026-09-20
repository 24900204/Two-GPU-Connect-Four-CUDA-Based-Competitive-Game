#include <cuda_runtime.h>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

constexpr int ROWS=6, COLS=7, CELLS=42;
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)

__device__ int dirCount(const int*b,int r,int c,int dr,int dc,int p){
 int n=0; r+=dr;c+=dc;
 while(r>=0&&r<ROWS&&c>=0&&c<COLS&&b[r*COLS+c]==p){n++;r+=dr;c+=dc;}
 return n;
}
__device__ bool win(const int*b,int r,int c,int p){
 const int d[4][2]={{1,0},{0,1},{1,1},{1,-1}};
 for(int i=0;i<4;i++) if(1+dirCount(b,r,c,d[i][0],d[i][1],p)+dirCount(b,r,c,-d[i][0],-d[i][1],p)>=4)return true;
 return false;
}
__global__ void evaluate(const int*b,int p,int*s){
 int c=threadIdx.x; if(c>=COLS)return; s[c]=-1000000;
 int r=ROWS-1; while(r>=0&&b[r*COLS+c])r--;
 if(r<0)return;
 int local[CELLS]; for(int i=0;i<CELLS;i++)local[i]=b[i];
 local[r*COLS+c]=p;
 if(win(local,r,c,p)){s[c]=10000;return;}
 int score=(3-abs(c-3))*20, opp=3-p;
 int rr=ROWS-1; while(rr>=0&&b[rr*COLS+c])rr--;
 if(rr>=0){int tmp[CELLS];for(int i=0;i<CELLS;i++)tmp[i]=b[i];tmp[rr*COLS+c]=opp;if(win(tmp,rr,c,opp))score+=7000;}
 s[c]=score;
}
struct Board{
 std::array<int,CELLS> a{};
 int drop(int c,int p){for(int r=ROWS-1;r>=0;r--)if(!a[r*COLS+c]){a[r*COLS+c]=p;return r;}return -1;}
 bool full()const{for(int c=0;c<COLS;c++)if(!a[c])return false;return true;}
 bool winner(int p)const{const int d[4][2]={{1,0},{0,1},{1,1},{1,-1}};for(int r=0;r<ROWS;r++)for(int c=0;c<COLS;c++)if(a[r*COLS+c]==p)for(auto&q:d){int n=1,R=r+q[0],C=c+q[1];while(R>=0&&R<ROWS&&C>=0&&C<COLS&&a[R*COLS+C]==p){n++;R+=q[0];C+=q[1];}if(n>=4)return true;}return false;}
 void print()const{for(int r=0;r<ROWS;r++){for(int c=0;c<COLS;c++)std::cout<<(a[r*COLS+c]==1?'X':a[r*COLS+c]==2?'O':'.')<<' ';std::cout<<'\n';}std::cout<<"---------------\n0 1 2 3 4 5 6\n\n";}
};
int choose(const Board&b,int p,int dev){
 CHECK(cudaSetDevice(dev));int *db,*ds;CHECK(cudaMalloc(&db,sizeof(int)*CELLS));CHECK(cudaMalloc(&ds,sizeof(int)*COLS));
 CHECK(cudaMemcpy(db,b.a.data(),sizeof(int)*CELLS,cudaMemcpyHostToDevice));
 evaluate<<<1,COLS>>>(db,p,ds);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 std::array<int,COLS>s{};CHECK(cudaMemcpy(s.data(),ds,sizeof(int)*COLS,cudaMemcpyDeviceToHost));cudaFree(db);cudaFree(ds);
 int best=-1,bs=-1000001;for(int c=0;c<COLS;c++)if(s[c]>bs){bs=s[c];best=c;}return best;
}
void worker(Board b,int p,int dev,int&move){move=choose(b,p,dev);}
int main(int argc,char**argv){
 int g0=0,g1=1;for(int i=1;i<argc;i++){std::string x=argv[i];if(x=="--gpu0"&&i+1<argc)g0=std::atoi(argv[++i]);else if(x=="--gpu1"&&i+1<argc)g1=std::atoi(argv[++i]);}
 int n=0;CHECK(cudaGetDeviceCount(&n));if(n<2){std::cerr<<"Two-GPU execution requires at least 2 CUDA devices. Detected "<<n<<".\n";return 2;}
 if(g0<0||g1<0||g0>=n||g1>=n||g0==g1){std::cerr<<"Choose two different valid GPU IDs.\n";return 2;}
 cudaDeviceProp p0{},p1{};CHECK(cudaGetDeviceProperties(&p0,g0));CHECK(cudaGetDeviceProperties(&p1,g1));
 std::cout<<"Player X -> GPU "<<g0<<" ("<<p0.name<<")\nPlayer O -> GPU "<<g1<<" ("<<p1.name<<")\n\n";
 Board b;int p=1;
 for(int turn=1;turn<=42&&!b.full();turn++){
  int mv=-1,dev=p==1?g0:g1;std::cout<<"Turn "<<turn<<": Player "<<(p==1?'X':'O')<<" using GPU "<<dev<<"\n";
  std::thread t(worker,b,p,dev,std::ref(mv));t.join();
  if(mv<0||b.drop(mv,p)<0)return 3;b.print();
  if(b.winner(p)){std::cout<<"Player "<<(p==1?'X':'O')<<" wins!\n";return 0;}p=3-p;
 }
 std::cout<<"Game ended in a draw.\n";
}
