#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>
#include "geometry.cuh"
#include "hand.cuh"

#define MAXTHREADS 512
bool refresh=true;

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

int ceil(int num, int den){
	return (num+den-1)/den;
}

void printArray(float4* arr, int n){
	for(int i=0; i<n; i++){
		printf("%f\t %f\t %f\t %f\n", arr[i].x, arr[i].y, arr[i].z, arr[i].w);
	}
}

void launch_kernel(float4 *d_pos, float4 *d_norm, uchar4 *d_color, dim3 mesh, float time){
	static const int n=mesh.x*mesh.y*mesh.z;
	static float4 *skel=new float4[25], *hand=new float4[n];
	
	if(refresh){
		skeleton(skel, angle);
		volume(hand, skel, mesh.y);
		
		checkCudaErrors(cudaMemcpy(d_pos, hand, n*sizeof(float4), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemset(d_color, 255u, n*sizeof(uchar4)));
		refresh=false;
	}
}