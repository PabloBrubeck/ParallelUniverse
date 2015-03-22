#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>

#define PI 3.14159265f

inline __host__ __device__ 
void cartesian(float4 &p, float x, float y, float z){
	p.x=x;
	p.y=y;
	p.z=z;
	p.w=1.f;
}
inline __host__ __device__ 
void cylindrical(float4 &p, float r, float theta, float z){
	p.x=r*cosf(theta);
	p.y=r*sinf(theta);
	p.z=z;
	p.w=1.f;
}
inline __host__ __device__ 
void spherical(float4 &p, float rho, float theta, float phi){
	float r=rho*sinf(phi);
	p.x=r*cosf(theta);
	p.y=r*sinf(theta);
	p.z=rho*cosf(phi);
	p.w=1.f;
}
inline __host__ __device__ 
float4 cross(float4 &a, float4 &b){
	return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0.f);
}

__global__
void normals(float4* d_normals, float4 *d_vertex, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;

		int a=(k*mesh.y+(j-1)%mesh.y)*mesh.x+i;
		int b=(k*mesh.y+j)*mesh.x+(i-1)%mesh.x;
		int c=(k*mesh.y+(j+1)%mesh.y)*mesh.x+i;
		int d=(k*mesh.y+j)*mesh.x+(i+1)%mesh.x;
		
		float4 CA=d_vertex[a]-d_vertex[c];
		float4 DB=d_vertex[b]-d_vertex[d];
		d_normals[gid]=cross(CA, DB);
	}
}
__global__
void colors(uchar4 *d_color, float4 *d_vertex, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		float4 *p=&d_vertex[gid];
		float x=p->x, y=p->y, z=p->z;
		float x2=x*x, y2=y*y, z2=z*z;
		float m2=x2+y2+z2;
		
		float r=(x<0?0:x2)+(y<0?y2:0)+(z<0?z2:0);
		float g=(x<0?x2:0)+(y<0?0:y2)+(z<0?z2:0);
		float b=(x<0?x2:0)+(y<0?y2:0)+(z<0?0:z2);
		
		d_color[gid].x=(unsigned char)min(255.f*r/m2, 255.f);
		d_color[gid].y=(unsigned char)min(255.f*g/m2, 255.f);
		d_color[gid].z=(unsigned char)min(255.f*b/m2, 255.f);
		d_color[gid].w=255u;
	}
}


__global__
void sphere(float4 *d_vertex, dim3 mesh, float r){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		float u=((2*i+1)*PI)/mesh.x;
		float v=(j*PI)/mesh.y;
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
		float u=(2*i*PI)/mesh.y;
		float v=(2*j*PI)/mesh.x;
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
		float t=(2*i*PI)/mesh.x;
		float s=w*((2*j)/(mesh.y-1.f)-1.f);
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
		float u=(2*i*PI)/mesh.x;
		float v=(2*j*PI)/mesh.y;
		float a=cosf(u/2);
		float b=sinf(u/2);
		float c=sinf(v);
		float d=sinf(2*v);
		cylindrical(d_vertex[gid], r*(w+a*c-b*d), u, r*(a*d+b*c));
	}
}
__global__
void catenoid(float4 *d_vertex, dim3 mesh, float c, float h){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		float u=((2*i+1)*PI)/mesh.x;
		float z=h*(1.f-(2*j)/(mesh.y-1.f));
		cylindrical(d_vertex[gid], c*coshf(z/c), u, z);
	}
}
__global__
void weirdThing(float4 *d_vertex, dim3 mesh, float w){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		float u=(2*j*PI)/mesh.y;
		float v=(2*i*PI)/mesh.x;
		float p=1.25f+sinf(3*u);
		cylindrical(d_vertex[gid], -w*(p*sinf(u-3*v)-6.f), v, -w*p*cosf(u-3*v));
	}
}