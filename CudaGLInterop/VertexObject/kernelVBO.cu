#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <helper_cuda.h>
#include <curand_kernel.h>
#include "Planck.h"

#define MAXTHREADS 512

static const float dvmax=1.f;
static bool first=true;

float *d_mass, *d_aux;
float3 *d_vel, *d_acc;

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
	const float x=u.x;
	const float y=u.y;
	const float z=u.z;
    return x*x+y*y+z*z;
}
__device__
float3 distance(const float4 &p, const float4 &q){
	return make_float3(q.x-p.x, q.y-p.y, q.z-p.z);
}
__device__
void bodyBodyInteraction(float3 &a, const float4 &p, const float4 &q, const float m){
	float3 r=distance(p, q);
	float r2=magnitude2(r)+epsilon;
	float w2=G*m*invsqrt(r2*r2*r2);
	a.x+=r.x*w2;
	a.y+=r.y*w2;
	a.z+=r.z*w2;
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
void generateSeed(curandState *d_states, unsigned long seed, const size_t n){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
	if(tid<n){
		curand_init(seed, tid, 0, &d_states[tid]);
	}
} 
__global__
void interact(float *d_mass, float4 *d_pos, float3 *d_acc, const size_t n){
	extern __shared__ float4 s_pos[];

	int tid=threadIdx.x;
	int src=blockIdx.x*blockDim.x+tid;
	int dst=blockIdx.y*blockDim.x+tid;
	
	s_pos[tid]=src<n? d_pos[src]: make_float4(0.f, 0.f, 0.f, 0.f);
	s_pos[tid].w=src<n? d_mass[src]: 0.f;
	__syncthreads();

	float4 pos=d_pos[dst];
	float3 acc=make_float3(0.f, 0.f, 0.f);
	for(int i=0; i<blockDim.x; i++){
		bodyBodyInteraction(acc, pos, s_pos[i], s_pos[i].w);
	}

	atomicAdd(&(d_acc[dst].x), acc.x);
	atomicAdd(&(d_acc[dst].y), acc.y);
	atomicAdd(&(d_acc[dst].z), acc.z);
	
}
__global__
void move(float4 *d_pos, float3 *d_vel, float3 *d_acc, float dt, const size_t n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		float vx=d_vel[i].x+d_acc[i].x*dt;
		float vy=d_vel[i].y+d_acc[i].y*dt;
		float vz=d_vel[i].z+d_acc[i].z*dt;
		d_pos[i].x+=vx*dt;
		d_pos[i].y+=vy*dt;
		d_pos[i].z+=vz*dt;
		d_vel[i]={vx, vy, vz};
		d_acc[i]={0.f, 0.f, 0.f};
	}
}
__global__
void initialState(float* d_mass, uchar4* d_color, float4 *d_pos, float3 *d_vel, 
	float3 *d_acc, curandState *d_states, uint2 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<mesh.x && j<mesh.y){
		int k=j*mesh.x+i;
		int n=mesh.x*mesh.y;

		curandState localState=d_states[k];
				
		float pi=3.1415926535f;
		float theta=2*k*pi/mesh.x;
		float r=mod((9*pi*k)/(n-1), 1.f);
		float z=(2*k>n? 1:-1)*exp(-r*r*5)/20;
		float x=r*cos(theta);
		float y=r*sin(theta);
		
		float m=2.f-r;
		float M=1.5f*n*r;
		float w=sqrt(G*M);

		d_pos[k]=make_float4(x, y, z, 1.f);
		d_vel[k]=make_float3(y*w, -x*w, 0.f);
		d_acc[k]=make_float3(0.f, 0.f, 0.f);
		d_mass[k]=m;

		float temp=1000.f+(m-1)*10000.f;
		d_color[k]=planckColor(temp);
		d_states[k]=localState;	
	}
}

int divideCeil(int num, int den){
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

void init(float4* d_pos, uchar4* d_color, const uint2 mesh){
	size_t n=mesh.x*mesh.y;
	curandState* d_states;
    checkCudaErrors(cudaMalloc((void**)&d_states, n*sizeof(curandState)));
	checkCudaErrors(cudaMalloc((void**)&d_aux, n*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_mass, n*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_vel, n*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&d_acc, n*sizeof(float3)));
	dim3 block(16, 16);
	dim3 grid(divideCeil(mesh.x, block.x), divideCeil(mesh.y, block.y));
	generateSeed<<<grid, block>>>(d_states, time(NULL), n);
	initialState<<<grid, block>>>(d_mass, d_color, d_pos, d_vel, d_acc, d_states, mesh);
}


// Wrapper for the __global__ call that sets up the kernel call
extern "C" 
void launch_kernel(float4 *d_pos, uchar4 *d_color, uint2 mesh, float time){
	if(first){
		init(d_pos, d_color, mesh);
		first=false;
	}

	size_t n=mesh.x*mesh.y;
	int block1D=MAXTHREADS;
	int grid1D=divideCeil(n, block1D);
	
	int p=128;
	int bytes=p*sizeof(float4);
	dim3 grid2D(divideCeil(n, p), divideCeil(n, p));
	interact<<<grid2D, p, bytes>>>(d_mass, d_pos, d_acc, n);

	mapMagnitude2<<<grid1D, block1D>>>(d_acc, d_aux, n);
	float dt=dvmax/sqrt(getMax(d_aux, n));
	
	move<<<grid1D, block1D>>>(d_pos, d_vel, d_acc, dt, n);
}