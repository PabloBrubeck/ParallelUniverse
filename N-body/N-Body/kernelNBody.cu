#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>
#include "geometry.h"
#include "linalg.h"
#include "Planck.h"

#define MAXTHREADS 512


__device__
struct Particle{
	float mass;
	float3 pos;
	float3 vel;
	float3 acc;
};


// device functions
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
void bodyBodyInteraction(float3 &acc, float3 &p, float4 &q){
	float3 r=make_float3(q.x-p.x, q.y-p.y, q.z-p.z);
	float r2=dot(r, r);
	float w2=q.w*invsqrt(r2*r2*r2+0.0006f);
	acc+=r*w2;
}


// simulation kernels
__global__
void initialState(uchar4 *d_color, Particle *d_body, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	
	if(i<mesh.x && j<mesh.y){
		int n=mesh.x*mesh.y;
		int k=j*mesh.x+i;

		float theta=2*i*PI/mesh.x;
		float r=float(j+1)/mesh.y;
		

		float x=r*cos(theta);
		float y=r*sin(theta);
		float z=0.f;
		
		float r0=1.f/3.f;
		
		float dr=1.f/mesh.y;
		float r1=r-dr/2;
		float r2=r+dr/2;
		

		float m=200000*r0*((r0+r2)*exp(-r1/r0)-(r0+r1)*exp(-r2/r0))/mesh.x;


		float v2=200000*r0*(r0-(r0+r)*exp(-r/r0));
		float w=sqrt(v2)/r;

		d_body[k].mass=m;
		d_body[k].pos={x, y, z};
		d_body[k].vel={y*w, -x*w, 0.f};
		d_body[k].acc={0.f, 0.f, 0.f};
		
		float temp=1000.f*m;
		d_color[k]=planckColor(temp);
	}
}
__global__
void interact(Particle *d_body, int n){
	extern __shared__ float4 s_buff[];
	int tid=threadIdx.x;
	int src=blockIdx.x*blockDim.x+tid;
	int dst=blockIdx.y*blockDim.x+tid;
	if(src<n){
		s_buff[tid]=make_float4(d_body[src].pos, d_body[src].mass);
	}
	__syncthreads();
	if(dst<n){
		float3 pos=d_body[dst].pos;
		float3 acc=make_float3(0.f, 0.f, 0.f);
		for(int i=0; i<blockDim.x; i++){
			bodyBodyInteraction(acc, pos, s_buff[i]);
		}
		atomicAdd(&(d_body[dst].acc.x), acc.x);
		atomicAdd(&(d_body[dst].acc.y), acc.y);
		atomicAdd(&(d_body[dst].acc.z), acc.z);
	}
}
__global__
void integrate(Particle *d_body, float dt, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		float3 vel=d_body[i].vel+d_body[i].acc*dt;
		d_body[i].acc=make_float3(0.f, 0.f, 0.f);
		d_body[i].vel=vel;
		d_body[i].pos+=vel*dt;
	}
}
__global__
void updatePoints(float4 *d_pos, Particle *d_body, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_pos[i]=make_float4(d_body[i].pos, 1.f);
	}
}


// auxiliary kernels
__global__
void mapMagnitude2(Particle *d_body, float *d_abs, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		float3 acc=d_body[i].acc;
		d_abs[i]=dot(acc, acc);
	}
}
__global__
void reduceMax(float *d_in, float *d_out, int n){   
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


// host functions
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


// kernel invocator
void launch_kernel(float4 *d_pos, float4 *d_norm, uchar4 *d_color, uint4 *d_index, dim3 mesh, float time){
	static const int n=mesh.x*mesh.y*mesh.z;
	static const int block1D=MAXTHREADS;
	static const int grid1D=ceil(n, block1D);
	static const int p=128;
	static const int bytes=p*sizeof(float4);
	static const dim3 grid2D(ceil(n, p), ceil(n, p));

	static const float dvmax=1.f;
	static Particle *d_body=NULL;
	static float *d_aux=NULL;

	if(d_body==NULL){
		// initialization
		checkCudaErrors(cudaMalloc((void**)&d_aux, n*sizeof(float)));
		checkCudaErrors(cudaMalloc((void**)&d_body, n*sizeof(Particle)));
		dim3 tblock(16, 16);
		dim3 tgrid(ceil(mesh.x, tblock.x), ceil(mesh.y, tblock.y));
		initialState<<<tgrid, tblock>>>(d_color, d_body, mesh);
	}

	// main kernel sequence
	interact<<<grid2D, p, bytes>>>(d_body, n);
	mapMagnitude2<<<grid1D, block1D>>>(d_body, d_aux, n);
	float dt=dvmax/sqrt(getMax(d_aux, n));
	integrate<<<grid1D, block1D>>>(d_body, dt, n);
	updatePoints<<<grid1D, block1D>>>(d_pos, d_body, n);

	cudaThreadSynchronize();
	checkCudaErrors(cudaGetLastError());
}