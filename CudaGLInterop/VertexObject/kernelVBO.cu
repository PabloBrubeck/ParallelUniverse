#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>
#include "geometry.h"
#include "linalg.h"

#define MAXTHREADS 512

// graphics pipeline-related kernels
__global__
void indices(uint4* d_index, float4* d_normals, float4 *d_vertex, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		
		int ii=(i+1)%mesh.x;
		int jj=(j+1)%mesh.y;
		int a=(k*mesh.y+j)*mesh.x+i; 
		int b=(k*mesh.y+j)*mesh.x+ii; 
		int c=(k*mesh.y+jj)*mesh.x+ii; 
		int d=(k*mesh.y+jj)*mesh.x+i;
				
		float4 CA=d_vertex[a]-d_vertex[c];
		float4 DB=d_vertex[b]-d_vertex[d];
		d_normals[gid]=cross(CA, DB);
		d_index[gid]=make_uint4(a, b, c, d);
	}
}
__global__
void normals(float4* d_normals, float4 *d_vertex, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;

		int ii=(i+1)%mesh.x;
		int jj=(j+1)%mesh.y;
		int a=(k*mesh.y+j)*mesh.x+i; 
		int b=(k*mesh.y+j)*mesh.x+ii; 
		//int c=(k*mesh.y+jj)*mesh.x+ii; 
		int d=(k*mesh.y+jj)*mesh.x+i;
		
		float4 AB=d_vertex[b]-d_vertex[a];
		float4 AD=d_vertex[d]-d_vertex[a];
		d_normals[gid]=cross(AB, AD);
	}
}
__global__
void colors(uchar4 *d_color, float4 *d_normals, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		float4 *p=d_normals+gid;
		float x=p->x, y=p->y, z=p->z;
		float x2=x*x, y2=y*y, z2=z*z;
		float m2=x2+y2+z2;
		
		float r=(x<0?0:x2)+(y<0?y2:0)+(z<0?z2:0);
		float g=(x<0?x2:0)+(y<0?0:y2)+(z<0?z2:0);
		float b=(x<0?x2:0)+(y<0?y2:0)+(z<0?0:z2);
		
		d_color[gid].x=(unsigned char)(255.f*r/m2)&0xff;
		d_color[gid].y=(unsigned char)(255.f*g/m2)&0xff;
		d_color[gid].z=(unsigned char)(255.f*b/m2)&0xff;
		d_color[gid].w=192u;
	}
}
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

inline uint ceil(uint num, uint den){
	return (num+den-1u)/den;
}

void launch_kernel(float4 *d_pos, float4 *d_norm, uchar4 *d_color, uint4 *d_index, dim3 mesh, float time){
	static const int n=mesh.x*mesh.y*mesh.z;
	static const dim3 block(MAXTHREADS, 1, 1);
	static const dim3 grid(ceil(mesh.x, block.x), ceil(mesh.y, block.y), ceil(mesh.z, block.z));
	static float4 *d_sphere=NULL, *d_torus=NULL;
	static float last=-FLT_MAX;
	static bool shape=true;
	if(d_sphere==NULL){
		checkCudaErrors(cudaMalloc((void**)&d_sphere, n*sizeof(float4)));
		checkCudaErrors(cudaMalloc((void**)&d_torus, n*sizeof(float4)));
		torus<<<grid, block>>>(d_torus, mesh, 2.f, 1.f);
		pretzel<<<grid, block>>>(d_sphere, mesh, 0.3f);
		checkCudaErrors(cudaMemcpy(d_pos, d_sphere, n*sizeof(float4), cudaMemcpyDeviceToDevice));
		indices<<<grid, block>>>(d_index, d_norm, d_pos, mesh);
	}
	float elapsed=time-last;
	if(elapsed>=2.f){
		if(elapsed>=7.f){
			shape=!shape;
			last=time;
		}else{
			animate<<<grid, block>>>(d_pos, shape? d_sphere:d_torus, mesh);
		}
		normals<<<grid, block>>>(d_norm, d_pos, mesh);
		colors<<< grid, block>>>(d_color, d_norm, mesh);
	}
	cudaThreadSynchronize();
	checkCudaErrors(cudaGetLastError());
}