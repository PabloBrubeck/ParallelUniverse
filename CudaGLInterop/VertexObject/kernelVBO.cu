#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>
#include "geometry.cuh"

#define MAXTHREADS 512
#define PI 3.14159265f

static dim3 grid3D, block3D;

__global__
void animate(uchar4* d_color, float4 *d_pos, dim3 mesh, float time){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;

	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		
		float s=param(-0.5f, 0.5f, 3, i, mesh.x);
		float t=param(0.f, 2*PI, 1, j, mesh.y);

		float4 temp={0.f,0.f,0.f,1.f};
		mobius(temp, s, t, 1.f);
		d_pos[gid]=0.99f*d_pos[gid]+0.01f*temp;
	}
}
__global__
void initialState(uchar4* d_color, float4 *d_pos, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;

	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;

		
		float u=param(0, 2*PI, 1, j, mesh.y);
		float v=param(0, 2*PI, 1, i, mesh.x);

		d_pos[gid].w=1.f;
		torus(d_pos[gid], u, v, 0.5f, 1.f);
		d_color[gid]={255u, 255u, 255u, 255u};
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
	if(mod(time,10)<=0.01f){
		block3D=dim3(8, 8, 8);
		grid3D=dim3(ceil(mesh.x, 8), ceil(mesh.y, 8), ceil(mesh.z, 8));
		initialState<<<grid3D, block3D>>>(d_color, d_pos, mesh);
	}
	animate<<<grid3D, block3D>>>(d_color, d_pos, mesh, time);
}