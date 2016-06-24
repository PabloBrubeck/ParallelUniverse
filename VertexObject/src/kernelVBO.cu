#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>
#include "geometry.h"
#include "linalg.h"

#define MAXTHREADS 512


__global__
void indices(uint4* d_index, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		
		int ii=(i+1)%mesh.x;
		int jj=(j+1)%mesh.y;
		d_index[gid].x=(k*mesh.y+j)*mesh.x+i;
		d_index[gid].y=(k*mesh.y+j)*mesh.x+ii;
		d_index[gid].z=(k*mesh.y+jj)*mesh.x+ii;
		d_index[gid].w=(k*mesh.y+jj)*mesh.x+i;
	}
}
__global__
void normalMapping(uchar4 *d_color, float4 *d_norm, float4* d_pos, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;

		int ii=(i+1)%mesh.x;
		int jj=(j+1)%mesh.y;
		
		float4 A=d_pos[gid];
		float4 T=d_pos[(k*mesh.y+j)*mesh.x+ii]-A;
		float4 B=d_pos[(k*mesh.y+jj)*mesh.x+i]-A;
		float4 N=normalize(cross(T, B));
		d_norm[gid]=N;

		float x=N.x,  y=N.y,  z=N.z;
		float x2=x*x, y2=y*y, z2=z*z;
		
		float r=(x<0?0:x2)+(y<0?y2:0)+(z<0?z2:0);
		float g=(x<0?x2:0)+(y<0?0:y2)+(z<0?z2:0);
		float b=(x<0?x2:0)+(y<0?y2:0)+(z<0?0:z2);
		
		d_color[gid].x=(unsigned char)(255.f*r)&0xff;
		d_color[gid].y=(unsigned char)(255.f*g)&0xff;
		d_color[gid].z=(unsigned char)(255.f*b)&0xff;
		d_color[gid].w=192u;
	}
}
__global__
void transform(float4 *d_pos, float4 *d_shape, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		d_pos[gid]=lerp(d_pos[gid], d_shape[gid], 0.01f);
		d_pos[gid].w=1.f;
	}
}



inline uint ceil(uint num, uint den){
	return (num+den-1)/den;
}

void morph(float4 *d_pos, float4 *d_norm, uchar4 *d_color, uint4 *d_index, dim3 mesh, float time){
	static const int n=mesh.x*mesh.y*mesh.z;
	static const dim3 block(MAXTHREADS, 1, 1);
	static const dim3 grid(ceil(mesh.x, block.x), ceil(mesh.y, block.y), ceil(mesh.z, block.z));
	static float4 *d_models=NULL;
	static float last=-1000.f;
	static bool shape=true;
	if(d_models==NULL){
		checkCudaErrors(cudaMalloc((void**)&d_models, 2*n*sizeof(float4)));
		torus<<<grid, block>>>(d_models, mesh, 2.f, 1.f);
		pretzel<<<grid, block>>>(d_models+n, mesh, 0.35f);
		checkCudaErrors(cudaMemcpy(d_pos, d_models, n*sizeof(float4), cudaMemcpyDeviceToDevice));
		indices<<<grid, block>>>(d_index, mesh);
		normalMapping<<<grid, block>>>(d_color, d_norm, d_pos, mesh);
	}
	float elapsed=time-last;
	if(elapsed>=2.f){
		if(elapsed>=7.f){
			shape=!shape;
			last=time;
		}
		transform<<<grid, block>>>(d_pos, d_models+(shape?0:n), mesh);
		normalMapping<<<grid, block>>>(d_color, d_norm, d_pos, mesh);
	}
	cudaThreadSynchronize();
	checkCudaErrors(cudaGetLastError());
}
void harmonic(float4 *d_pos, float4 *d_norm, uchar4 *d_color, uint4 *d_index, dim3 mesh, float time){
	static const dim3 block(MAXTHREADS, 1, 1);
	static const dim3 grid(ceil(mesh.x, block.x), ceil(mesh.y, block.y), ceil(mesh.z, block.z));

	if(time==0){
		int l=4, m=2;
		float w=0.1;

		float* d_rho;
		checkCudaErrors(cudaMalloc((void**)&d_rho, mesh.x*mesh.y*sizeof(float)));
		checkCudaErrors(cudaMemset(d_rho, 0.f, mesh.x*mesh.y*sizeof(float)));
		sphericalHarmonic<<<grid, block>>>(d_rho, mesh, l, m, w);
		sphericalPlot<<<grid, block>>>(d_pos, mesh, d_rho);
		indices<<<grid, block>>>(d_index, mesh);
		normalMapping<<<grid, block>>>(d_color, d_norm, d_pos, mesh);
	}
	cudaThreadSynchronize();
	checkCudaErrors(cudaGetLastError());
}

void launch_kernel(float4 *d_pos, float4 *d_norm, uchar4 *d_color, uint4 *d_index, dim3 mesh, float time){
	morph(d_pos, d_norm, d_color, d_index, mesh, time);
}
