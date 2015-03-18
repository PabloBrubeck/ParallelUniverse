#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>
#include "geometry.cuh"

#define MAXTHREADS 512
#define PI 3.14159265f

float4* d_shape;

__global__
void animate(float4 *d_pos, float4 *d_shape, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		d_pos[gid]=0.99f*d_pos[gid]+0.01f*d_shape[gid];
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


void launch_kernel(float4 *d_pos, uchar4 *d_color, dim3 mesh, float time){
	static int n=mesh.x*mesh.y*mesh.z;
	static dim3 block(8, 8, 8);
	static dim3 grid(ceil(mesh.x, 8), ceil(mesh.y, 8), ceil(mesh.z, 8));
	
	if(time==0.f){
		cudaMalloc((void**)&d_shape, n*sizeof(float4));
		figureEight<<<block, grid>>>(d_shape, mesh, 1.f, 2.f);
	}
	if(mod(time, 10)<=0.01f){
		torus<<<block, grid>>>(d_pos, mesh, 1.f, 2.f);
	}
	animate<<<grid, block>>>(d_pos, d_shape, mesh);
	color<<<grid, block>>>(d_color, d_pos, mesh);
}