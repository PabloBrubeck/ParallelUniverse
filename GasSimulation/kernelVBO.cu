#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <helper_cuda.h>
#include <curand_kernel.h>
#include <cutil_math.h>
#include "Planck.h"

#define L 0.015625f               // cell lenght = 2*particle radius
#define L2 L*L                    // cell lenght squared
#define cells 129                 // number of cells in one dimension
#define cells3 cells*cells*cells  // number of cells in three dimensions
#define MAXTHREADS 512

typedef unsigned int uint;
struct Particle{
	float3 pos;
	float3 vel;
	float3 acc;
};


Particle *d_gas;
uint *d_gridCounters, *d_gridCells;
static bool first=true;


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
float quadFormula(float a, float b, float c){
	return (-b+sqrtf(b*b-4.f*a*c))/(2.f*a);
}
__device__ 
void collision(Particle &a, Particle &b){
	float3 ds=a.pos-b.pos;
	float3 dv=a.vel-b.vel;
	float A=dot(dv, dv);
	float B=-2.f*dot(ds, dv);
	float C=dot(ds, ds)-L2;
	float t0=quadFormula(A,B,C);
	float3 r=ds-dv*t0; // |r|=L always

	dv=r*(dot(dv, r)/L2);
	a.vel+=dv;
	b.vel-=dv;
	a.pos+=dv*t0;
	b.pos-=dv*t0;
}
__device__
int cellIndex(float3 &pos){
	int i=(int)((pos.x+1.f)/L);
	int j=(int)((pos.y+1.f)/L);
	int k=(int)((pos.z+1.f)/L);
	return (k*cells+j)*cells+i;
}


__global__
void initialState(uchar4* d_color, float4* d_pos, Particle* d_gas, uint3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;

		float x=((float)(2*i))/mesh.x-1.f;
		float y=((float)(2*j))/mesh.y-1.f;
		float z=((float)(2*k))/mesh.z-1.f;
		
		d_pos[gid]={x, y, z, 1.f};
		d_gas[gid].pos={x, y, z};
		d_gas[gid].vel={x, y, z};
		d_gas[gid].acc={0.f, 0.f, 0.f};
		float temp=2500.f*dot(d_gas[gid].vel, d_gas[gid].vel);
		d_color[gid]=planckColor(temp);
	}
}
__global__
void updateGrid(Particle* d_gas, uint* d_gridCounters, uint* d_gridCells, int n){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	if(gid<n){
		int cid=cellIndex(d_gas[gid].pos);
		int s=atomicAdd(&(d_gridCounters[cid]), 1u);
		if(s<4u){
			d_gridCells[4*cid+s]=gid;
		}
	}
}
__global__
void neighbors(Particle* d_gas, uint* d_gridCounters, uint* d_gridCells){
	int hx=blockIdx.x*blockDim.x+threadIdx.x;
	int hy=blockIdx.y*blockDim.y+threadIdx.y;
	int hz=blockIdx.z*blockDim.x+threadIdx.z;
	if(hx<cells && hy<cells && hz<cells){
		int hid=(hz*cells+hy)*cells+hx;
		int hcount=d_gridCounters[hid];
		if(hcount==0){
			return;
		}
		int ncount=0;
		int* neighbors=new int[108];
		int index=0;

		int nx, ny, nz;
		for(int i=-1; i<=1; i++){
			nx=hx+i;
			for(int j=-1; j<=1; j++){
				ny=hy+j;
				for(int k=-1; k<=1; k++){					
					nz=hz+k;
					if(	(nx>=0 && nx<cells) && 
						(ny>=0 && ny<cells) && 
						(nz>=0 && nz<cells)){
						int nid=(nz*cells+ny)*cells+nx;
						int acount=d_gridCounters[nid];
						for(int m=0; m<acount; m++){
							neighbors[index++]=d_gridCells[4*nid+m];
						}
						ncount+=acount;
					}
				}
			}
		}

		Particle home, away;
		for(int h=0; h<hcount; h++){
			home=d_gas[d_gridCells[4*hid+h]];
			for(int a=0; a<ncount; a++){
				away=d_gas[neighbors[a]];
				// Check if particles are close enough
				float3 r=home.pos-away.pos;
				float r2=dot(r,r);
				if(r2<L2){
					// Check if the barycenter belongs to home
					float3 b=(home.pos+away.pos)/2.f;
					if(cellIndex(b)==hid){
						collision(home, away);
					}
				}
			}
		}		
	}
}
__global__ 
void move(uchar4* d_color, float4* d_pos, Particle* d_gas, float step, int n){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	if(gid<n){
		float3 vel=d_gas[gid].vel+d_gas[gid].acc*step;
		float3 pos=d_gas[gid].pos+vel*step;
		if(abs(pos.x)>1.f){
			vel.x=-vel.x;
		}
		if(abs(pos.y)>1.f){
			vel.y=-vel.y;
		}
		if(abs(pos.z)>1.f){
			vel.z=-vel.z;
		}
		pos=clamp(pos, -1.f, 1.f);
		d_pos[gid]={pos.x, pos.y, pos.z, 1.f};
		d_gas[gid].pos=pos;
		d_gas[gid].vel=vel;
		d_gas[gid].acc={0.f, 0.f, 0.f};
		float temp=2500.f*dot(vel,vel);
		d_color[gid]=planckColor(temp);
	}
}


int ceil(int num, int den){
	return (num+den-1)/den;
}
uint nextPowerOf2(uint n){
  uint k=0;
  if(n&&!(n&(n-1))){
	  return n;
  }
  while(n!=0){
    n>>=1;
    k++;
  }
  return 1<<k;
}
void printArray(uint* arr, int n){
	for(int i=0; i<n; i++){
		printf("%u ", arr[i]);
	}
	printf("\n");
}
void printFromDevice(uint* d_array, int length){
    uint *h_temp=new uint[length];
    checkCudaErrors(cudaMemcpy(h_temp, d_array, length*sizeof(uint), cudaMemcpyDeviceToHost)); 
    printArray(h_temp, length);
    delete[] h_temp;
}


void init(float4* d_pos, uchar4* d_color, uint3 mesh){
	int n=mesh.x*mesh.y*mesh.z;
	checkCudaErrors(cudaMalloc((void**)&d_gas, n*sizeof(Particle)));
	checkCudaErrors(cudaMalloc((void**)&d_gridCounters, cells3*sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&d_gridCells, 4*cells3*sizeof(uint)));
	dim3 grid3D(ceil(mesh.x, 8), ceil(mesh.y, 8), ceil(mesh.z, 8));
	initialState<<<grid3D, dim3(8, 8, 8)>>>(d_color, d_pos, d_gas, mesh);
}
void launch_kernel(float4 *d_pos, uchar4 *d_color, uint3 mesh, float time){
	if(first){
		init(d_pos, d_color, mesh);
		first=false;
	}
	int n=mesh.x*mesh.y*mesh.z;
	float step=0.001f;

	checkCudaErrors(cudaMemset(d_gridCounters, 0u, cells3*sizeof(uint)));
	checkCudaErrors(cudaMemset(d_gridCells, 0u, 4*cells3*sizeof(uint)));
	
	dim3 grid1D(ceil(n, MAXTHREADS));
	updateGrid<<<grid1D, MAXTHREADS>>>(d_gas, d_gridCounters, d_gridCells, n);
	move<<<grid1D, MAXTHREADS>>>(d_color, d_pos, d_gas, step, n);
	
}