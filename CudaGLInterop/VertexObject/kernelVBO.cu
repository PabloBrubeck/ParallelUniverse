#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <helper_cuda.h>
#include <curand_kernel.h>


#define MAXTHREADS 512

static const float dvmax=0.001f;
static bool first=true;

float *d_mass, *d_aux;
float3 *d_vel, *d_acc;


__device__
static const float G=0.0001f, epsilon=0.1f;
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
void set(float3 &u, const float x, const float y, const float z){
	u.x=x;
	u.y=y;
	u.z=z;
}
__device__
float magnitude2(const float3 &u){
	const float x=u.x;
	const float y=u.y;
	const float z=u.z;
    return x*x+y*y+z*z;
}
__device__
float3 distance(const float4& p, const float4& q){
	return make_float3(q.x-p.x, q.y-p.y, q.z-p.z);
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
void initialState(float* d_mass, uchar4* d_color, float4 *d_pos, float3 *d_vel, 
	float3 *d_acc, curandState *d_states, const size_t n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		int width=32;
		int height=32;
		int time=0;
		int x=i%width;
		int y=i/width;

		// calculate uv coordinates
		float u = x / (float) width;
		float v = y / (float) height;
		u = u*2.0f - 1.0f;
		v = v*2.0f - 1.0f;

		// calculate simple sine wave pattern
		float freq = 4.0f;
		float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

		// write output vertex
		d_pos[i] = make_float4(u, w, v, 1.f);

		//curandState localState=d_states[i];

		//d_pos[i].x=curand_normal(&localState);
		//d_pos[i].y=curand_normal(&localState);
		//d_pos[i].z=curand_normal(&localState);
		//d_pos[i].w=1.f;

		set(d_vel[i], 0.f, 0.f, 0.f);
		set(d_acc[i], 0.f, 0.f, 0.f);
		float m=1.f+4*w*w;
		d_mass[i]=m;
		d_color[i]=make_uchar4(255, 255, 255, 255);

		//d_states[i]=localState;
	}
}
__global__
void interact(float *d_mass, float4 *d_pos, float3 *d_acc, const size_t n){
	extern __shared__ float3 s_acc[];
	int tid=threadIdx.x;
	int i=blockIdx.x;
	int j=blockIdx.y*blockDim.x+tid;
	if(j>=n || i==j){
		set(s_acc[tid], 0.f, 0.f, 0.f);
	}else{
		float3 r=distance(d_pos[i], d_pos[j]);
		float r2=magnitude2(r)+epsilon;
		float w2=G*d_mass[j]*invsqrt(r2*r2*r2);
		set(s_acc[tid], r.x*w2, r.y*w2, r.z*w2);
	}
	// Reduction
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid<s){
            s_acc[tid].x+=s_acc[tid+s].x;
			s_acc[tid].y+=s_acc[tid+s].y;
			s_acc[tid].z+=s_acc[tid+s].z;
        }
        __syncthreads();
    }
	if(tid==0){
		atomicAdd(&(d_acc[i].x), s_acc[0].x);
		atomicAdd(&(d_acc[i].y), s_acc[0].y);
		atomicAdd(&(d_acc[i].z), s_acc[0].z);
    }
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
		set(d_vel[i], vx, vy, vz);
		set(d_acc[i], 0.f, 0.f, 0.f);
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

void init(float4* d_pos, uchar4* d_color, const size_t n){
	curandState* d_states;
    checkCudaErrors(cudaMalloc((void**)&d_states, n*sizeof(curandState)));
	checkCudaErrors(cudaMalloc((void**)&d_aux, n*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_mass, n*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_vel, n*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&d_acc, n*sizeof(float3)));
	int block1D=MAXTHREADS;
	int grid1D=divideCeil(n, block1D);
	generateSeed<<<grid1D, block1D>>>(d_states, 1234567890, n);
	initialState<<<grid1D, block1D>>>(d_mass, d_color, d_pos, d_vel, d_acc, d_states, n);
}


// Wrapper for the __global__ call that sets up the kernel call
extern "C" 
void launch_kernel(float4* d_pos, uchar4* d_color, 
	unsigned int mesh_width, unsigned int mesh_height, float time){
	const size_t n=mesh_width*mesh_height;
	if(first){
		init(d_pos, d_color, n);
		first=false;
	}

	int block1D=MAXTHREADS;
	int grid1D=divideCeil(n, block1D);
	int bytes=block1D*sizeof(float3);
	dim3 block2D(MAXTHREADS);
	dim3 grid2D(n, divideCeil(n, MAXTHREADS));
	
	interact<<<grid2D, block2D, bytes>>>(d_mass, d_pos, d_acc, n);

	mapMagnitude2<<<grid1D, block1D>>>(d_acc, d_aux, n);
	float dt=dvmax/sqrt(getMax(d_aux, n));
	
	move<<<grid1D, block1D>>>(d_pos, d_vel, d_acc, dt, n);
}










__global__ 
void kernel(float4* pos, uchar4 *colorPos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
 
    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;
 
    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;
 
    // write output vertex
    pos[y*width+x] = make_float4(u, w, v, 1.0f);
    colorPos[y*width+x].w = 0;
    colorPos[y*width+x].x = 255.f *0.5*(1.f+sinf(w+x));
    colorPos[y*width+x].y = 255.f *0.5*(1.f+sinf(x)*cosf(y));
    colorPos[y*width+x].z = 255.f *0.5*(1.f+sinf(w+time/10.f));
}