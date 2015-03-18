#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>

#define PI 3.14159265f


__device__
void cylindrical(float4 &p, float r, float theta, float z){
	p.x=r*cosf(theta);
	p.y=r*sinf(theta);
	p.z=z;
	p.w=1.f;
}
__device__
void spherical(float4 &p, float rho, float theta, float phi){
	float r=rho*sinf(phi);
	p.x=r*cosf(theta);
	p.y=r*sinf(theta);
	p.z=rho*cosf(phi);
	p.w=1.f;
}


__global__
void color(uchar4 *d_color, float4 *d_vertex, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		float4 p=d_vertex[gid];
		float x2=p.x*p.x;
		float y2=p.y*p.y;
		float z2=p.z*p.z;
		float r2=x2+y2+z2;
		p={x2/r2, y2/r2, z2/r2, 1.f};
		d_color[gid]={255*p.x, 255*p.y, 255*p.z, 255};
	}
}
__global__
void sphere(float4 *d_vertex, dim3 mesh, float r){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		float u=(2*j*PI)/mesh.y;
		float v=(2*i*PI)/mesh.x;
		spherical(d_vertex[gid], r, u, v);
	}
}
__global__
void torus(float4 *d_vertex, dim3 mesh, float a, float c){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		float u=(2*j*PI)/mesh.y;
		float v=(2*i*PI)/mesh.x;
		cylindrical(d_vertex[gid], c+a*cosf(v), u, a*sinf(v));
	}
}
__global__
void mobius(float4 *d_vertex, dim3 mesh, float r, float w){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		float s=w*((2*i)/(mesh.x-1.f)-1.f);
		float t=(2*j*PI)/mesh.y;
		cylindrical(d_vertex[gid], r+s*cosf(t/2.f), t, s*sinf(t/2.f));
	}
}
__global__
void figureEight(float4 *d_vertex, dim3 mesh, float r, float w){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		float v=(2*i*PI)/mesh.x;
		float u=(2*j*PI)/mesh.y;
		float a=cosf(u/2);
		float b=sinf(u/2);
		float c=sinf(v);
		float d=sinf(2*v);
		cylindrical(d_vertex[gid], r*(w+a*c-b*d), u, r*(a*d+b*c));
	}
}