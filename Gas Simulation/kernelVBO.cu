#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <helper_cuda.h>
#include <curand_kernel.h>
#include "Planck.h"

#define MAXTHREADS 512

static bool first=true;

struct Particle{
	float3 vel;
	float3 acc;
};

Particle *d_gas;

__device__
static const float R2=0.00001f;

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
float dotP(float3 &u, float3 &v){
	return u.x*v.x+u.y*v.y+u.z*v.z;
}
__device__
float3 distance(float4 &p, float4 &q){
	return {p.x-q.x, p.y-q.y, p.z-q.z};
}
__device__
float3 subtract(float3 &u, float3 &v){
	return {u.x-v.x, u.y-v.y, u.z-v.z};
}
__device__
void atomicAdd3(float3 &u, float3 &v, bool right){
	if(right){
		atomicAdd(&(u.x), v.x);
		atomicAdd(&(u.y), v.y);
		atomicAdd(&(u.z), v.z);
	}else{
		atomicAdd(&(u.x), -v.x);
		atomicAdd(&(u.y), -v.y);
		atomicAdd(&(u.z), -v.z);
	}
}
__device__
void translate(float4 p, float3 r, bool right){
	if(right){
		p.x+=r.x;
		p.y+=r.y;
		p.z+=r.z;
	}else{
		p.x-=r.x;
		p.y-=r.y;
		p.z-=r.z;
	}
}

__global__
void initialState(uchar4* d_color, float4 *d_pos, Particle* d_gas, uint2 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	
	if(i<mesh.x && j<mesh.y){
		int k=j*mesh.x+i;
			
		float pi=3.1415926535f;
		float theta=2*i*pi/mesh.x;
		float phi=j*pi/(mesh.y-1);
		float r=sin(phi);
		float z=cos(phi);
		float x=r*cos(theta);
		float y=r*sin(theta);
		d_pos[k]={x, y, z, 1.f};
		d_gas[k].vel={-x, -y, -z};
		d_gas[k].acc={0.f, 0.f, 0.f};
		
		float temp=dotP(d_gas[i].vel,d_gas[i].vel)*2500.f;
		d_color[i]=planckColor(temp);
	}
}
__global__ 
void interact(uchar4* d_color, float4* d_pos, Particle* d_gas, int n){
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int j=blockDim.y*blockIdx.y+threadIdx.y;
	if(i<j && i<n && j<n){
		float3 r=distance(d_pos[i], d_pos[j]);
		float r2=dotP(r,r);
		if(r2<R2){
			float w=dotP(subtract(d_gas[i].vel, d_gas[j].vel), r)/r2;
			float3 dv={w*r.x, w*r.y, w*r.z};
			atomicAdd3(d_gas[i].acc, dv, true);
			atomicAdd3(d_gas[j].acc, dv, false);

			float h=0.5f*(invsqrt(r2/R2)-1.f);
			float3 hr={h*r.x, h*r.y, h*r.z};
			translate(d_pos[i], hr, true);
			translate(d_pos[j], hr, false);
		}
	}
}
__global__ 
void move(uchar4* d_color, float4* d_pos, Particle* d_gas, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		float dt=0.001f;
		float vx=d_gas[i].vel.x+d_gas[i].acc.x;
		float vy=d_gas[i].vel.y+d_gas[i].acc.y;
		float vz=d_gas[i].vel.z+d_gas[i].acc.z;

		float x=d_pos[i].x+vx*dt;
		float y=d_pos[i].y+vy*dt;
		float z=d_pos[i].z+vz*dt;
		if(abs(x)>1){
			vx=-vx;
			x=x>0?1:-1;
		}
		if(abs(y)>1){
			vy=-vy;
			y=y>0?1:-1;
		}
		if(abs(z)>1){
			vz=-vz;
			z=z>0?1:-1;
		}
		d_pos[i]={x, y, z, 1.f};
		d_gas[i].vel={vx, vy, vz};
		d_gas[i].acc={0.f, 0.f, 0.f};
		float temp=1000.f+dotP(d_gas[i].vel,d_gas[i].vel)*800.f;
		d_color[i]=planckColor(temp);
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

void init(float4* d_pos, uchar4* d_color, const uint2 mesh){
	size_t n=mesh.x*mesh.y;
	curandState* d_states;
    checkCudaErrors(cudaMalloc((void**)&d_states, n*sizeof(curandState)));
	checkCudaErrors(cudaMalloc((void**)&d_gas, n*sizeof(Particle)));
	dim3 block(16, 16);
	dim3 grid(divideCeil(mesh.x, block.x), divideCeil(mesh.y, block.y));
	initialState<<<grid, block>>>(d_color, d_pos, d_gas, mesh);
}


// Wrapper for the __global__ call that sets up the kernel call
void launch_kernel(float4 *d_pos, uchar4 *d_color, uint2 mesh, float time){
	if(first){
		init(d_pos, d_color, mesh);
		first=false;
	}
	size_t n=mesh.x*mesh.y;

	int block1D=__min(MAXTHREADS, nextPowerOf2(n));
	int grid1D=divideCeil(n,block1D);
	dim3 block2D(16, 16);
	dim3 grid2D(divideCeil(n, block2D.x), divideCeil(n, block2D.y));
	
	interact<<<grid2D, block2D>>>(d_color, d_pos, d_gas, n);
	move<<<grid1D, block1D>>>(d_color, d_pos, d_gas, n);
}