#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <helper_cuda.h>
#include <curand_kernel.h>
#include <cutil_math.h>
#include "Planck.h"

#define MAXTHREADS 512

static const float dvmax=1.f;
static bool first=true;

struct Particle{
	float mass;
	float3 pos;
	float3 vel;
	float3 acc;
};

Particle *d_stars;
float *d_aux;

__device__
static const float G=1.f, epsilon=0.0006f;

__device__
float mod(float x, float y) {
	return x-y*floor(x/y);
}
__device__
float invsqrt(float x){
	long i;
	float x2, y;
	const float threehalfs = 1.5F;
	x2=x*0.5F;
	y=x;
	i=*(long*)&y;                // evil floating point bit level hacking
	i=0x5f3759df-(i>>1);         // what the fuck?
	y=*(float*)&i;
	y=y*(threehalfs-(x2*y*y));   // 1st iteration
    y=y*(threehalfs-(x2*y*y));   // 2nd iteration, this can be removed
	return y;
}
__device__
float magnitude2(const float3 &u){
    return u.x*u.x+u.y*u.y+u.z*u.z;
}
__device__
float3 bodyBodyInteraction(const Particle &a, const Particle &b){
	float3 r=a.pos-b.pos;
	float r2=magnitude2(r);
	return (b.mass*invsqrt(r2*r2*r2+epsilon))*r;
}


__global__
void mapMagnitude2(float3 *d_vec, float *d_abs, const size_t n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_abs[i]=magnitude2(d_vec[i]);
	}
}
__global__
void reduceMax(float *d_in, float *d_out, const size_t n){   
    extern __shared__ float shared[];
	int tid=threadIdx.x;
    int gid=blockIdx.x*blockDim.x+tid;
	shared[tid]= gid<n? d_in[gid]: -FLT_MAX;
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid<s){
            shared[tid]=__max(shared[tid], shared[tid+s]);
        }
        __syncthreads();
    }
    if(tid==0){
        d_out[blockIdx.x]=shared[0];
    }
}
__global__
void interact(Particle *d_stars, const size_t n){
	extern __shared__ Particle s_par[];
	int tid=threadIdx.x;
	int src=blockIdx.x*blockDim.x+tid;
	int dst=blockIdx.y*blockDim.x+tid;
	if(src<n){
		s_par[tid]=d_stars[src];
	}
	__syncthreads();
	if(dst<n){
		float3 acc={0.f, 0.f, 0.f};
		for(int i=0; i<blockDim.x; i++){
			acc+=bodyBodyInteraction(d_stars[dst], s_par[i]);
		}
		atomicAdd(&(d_stars[dst].acc.x), acc.x);
		atomicAdd(&(d_stars[dst].acc.y), acc.y);
		atomicAdd(&(d_stars[dst].acc.z), acc.z);
	}
}
__global__
void integrate(float4 *d_pos, Particle* d_stars, float dt, const size_t n){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	if(gid<n){
		d_stars[gid].vel+=d_stars[gid].acc*dt;
		float3 pos=d_stars[gid].pos+d_stars[gid].vel*dt;
		d_stars[gid].pos=pos;
		d_stars[gid].acc={0.f, 0.f, 0.f};
		d_pos[gid]={pos.x, pos.y, pos.z, 1.f};
	}
}
__global__
void initialState(uchar4* d_color, float4 *d_pos, Particle* d_stars, uint3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	
	if(i<mesh.x && j<mesh.y){
		int n=mesh.x*mesh.y;
		int k=j*mesh.x+i;
			
		float PI=3.1415926535f;
		float theta=2*k*PI/mesh.x;
		float r=mod((9*PI*k)/(n-1), 1.f);
		float z=(2*k>n? 1:-1)*exp(-r*r*5)/20;
		float x=r*cos(theta);
		float y=r*sin(theta);
		
		float m=2.f-r;
		float M=1.5f*n*r;
		float w=sqrt(G*M);
		
		d_pos[k]={x, y, z, 1.f};
		d_stars[k].pos={x, y, z};
		d_stars[k].vel={y*w, -x*w, 0.f};
		d_stars[k].acc={0.f, 0.f, 0.f};
		d_stars[k].mass=m;

		float temp=1000.f+(m-1)*10000.f;
		d_color[k]=planckColor(temp);
	}
}

int ceil(int num, int den){
	return (num+den-1)/den;
}
unsigned int nextPowerOf2(unsigned int n){
  unsigned k=0;
  if(n&&!(n&(n-1))){
	  return n;
  }
  while(n!=0){
    n>>=1;
    k++;
  }
  return 1<<k;
}
float getMax(float *d_in, const size_t numElems){
	int n=numElems;
	int grid, block=MAXTHREADS;
	float *h_out=new float();
	do{
		grid=(n+block-1)/block;
		if(grid==1){
			block=nextPowerOf2(n);
		}
		reduceMax<<<grid, block, block*sizeof(float)>>>(d_in, d_in, n);
		n=grid;
	}while(grid>1);
	checkCudaErrors(cudaMemcpy(h_out, d_in, sizeof(float), cudaMemcpyDeviceToHost));
	return *h_out;
}

void init(float4* d_pos, uchar4* d_color, const uint3 mesh){
	size_t n=mesh.x*mesh.y*mesh.z;
	checkCudaErrors(cudaMalloc((void**)&d_aux, n*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_stars, n*sizeof(Particle)));
	dim3 block(16, 16);
	dim3 grid(ceil(mesh.x, block.x), ceil(mesh.y, block.y));
	initialState<<<grid, block>>>(d_color, d_pos, d_stars, mesh);
}
void launch_kernel(float4 *d_pos, uchar4 *d_color, uint3 mesh, float time){
	if(time==0.f){
		init(d_pos, d_color, mesh);
	}

	size_t n=mesh.x*mesh.y*mesh.z;
	int block1D=MAXTHREADS;
	int grid1D=ceil(n, block1D);
	
	int p=128;
	int bytes=p*sizeof(Particle);
	dim3 grid2D(ceil(n, p), ceil(n, p));
	interact<<<grid2D, p, bytes>>>(d_mass, d_pos, d_acc, n);
	
	mapMagnitude2<<<grid1D, block1D>>>(d_stars, d_aux, n);
	float dt=dvmax/sqrt(getMax(d_aux, n));
	
	integrate<<<grid1D, block1D>>>(d_pos, d_stars, dt, n);
}