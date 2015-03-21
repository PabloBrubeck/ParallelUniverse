#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>
#include "geometry.cuh"

#define MAXTHREADS 512

__global__
void animate(float4 *d_vertex, float4 *d_shape, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		d_vertex[gid]=0.99f*d_vertex[gid]+0.01f*d_shape[gid];
	}
}
__global__
void ricciFlow(float4 *d_vertex, float4 *d_normals, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		d_vertex[gid]+=0.01*d_normals[gid];
	}
}

int ceil(int num, int den){
	return (num+den-1)/den;
}
float mod(float x, float y){
	return x-y*floor(x/y);
}
unsigned nextPowerOf2(unsigned n){
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


void launch_kernel(float4 *d_vertex, uchar4 *d_color, dim3 mesh, float time){
	static const int n=mesh.x*mesh.y*mesh.z;
	static const dim3 block(8, 8, 8);
	static const dim3 grid(ceil(mesh.x, 8), ceil(mesh.y, 8), ceil(mesh.z, 8));
	static float4 *d_normals=NULL, *d_sphere=NULL, *d_torus=NULL;
	static float last=-20.f;
	static bool shape=true;

	if(d_normals==NULL){
		cudaMalloc((void**)&d_normals, n*sizeof(float4));
		cudaMalloc((void**)&d_sphere, n*sizeof(float4));
		cudaMalloc((void**)&d_torus, n*sizeof(float4));

		figureEight<<<grid, block>>>(d_sphere, mesh, 1.f, 2.f);
		weirdThing<<<grid, block>>>(d_torus, mesh, 0.3f);
		cudaMemcpy(d_vertex, d_sphere, n*sizeof(float4), cudaMemcpyDeviceToDevice);
	}

	float elapsed=time-last;
	if(elapsed>=2.f){
		if(elapsed>=7.f){
			shape=!shape;
			last=time;
		}else{
			animate<<<grid, block>>>(d_vertex, shape? d_sphere:d_torus, mesh);
		}
		normals<<<grid, block>>>(d_normals, d_vertex, mesh);
		colors<<< grid, block>>>(d_color, d_normals, mesh);
	}
}