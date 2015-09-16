#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>
#include "polynomial.h"

inline __device__
uint4 gridIdx(dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	return make_uint4(i, j, k, (k*mesh.y+j)*mesh.x+i);
}
inline __device__
bool inside(uint4 gid, dim3 mesh){
	return gid.x<mesh.x && gid.y<mesh.y && gid.z<mesh.z;
}


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

__global__ void unitTangent(float4 *d_uT, float4 *d_vertex, dim3 mesh){
	uint4 gid=gridIdx(mesh);
	if(inside(gid, mesh)){
		int a=((gid.x+mesh.x-1)%mesh.x)*mesh.y+gid.y;
		int b=((gid.x+1)%mesh.x)*mesh.y+gid.y;
		d_uT[gid.w]=normalize(d_vertex[b]-d_vertex[a]);
	}
}


__global__ void sphere(float4 *d_vertex, dim3 mesh, float r){
	uint4 gid=gridIdx(mesh);
	if(inside(gid, mesh)){
		float u=(M_PI*(2*gid.x+1))/mesh.x;
		float v=(float)gid.y/(mesh.y-1);
		float t=2.f*sqrtf(v*(1.f-v));
		float z=2.f*v-1.f;
		cylindrical(d_vertex[gid.w], r*t, u, r*z);
	}
}
__global__ void sphericalPlot(float4 *d_vertex, dim3 mesh, float* d_rho){
	uint4 gid=gridIdx(mesh);
	if(inside(gid, mesh)){
		float u=(M_PI*(2*gid.x+1))/mesh.x;
		float v=(float)gid.y/(mesh.y-1);
		float t=2.f*sqrtf(v*(1.f-v));
		float z=2.f*v-1.f;
		cylindrical(d_vertex[gid.w], t*d_rho[gid.w], u, z*d_rho[gid.w]);
	}
}
__global__ void sphericalHarmonic(float *d_rho, dim3 mesh, float* d_Pml, int m, int l, float scale){
	uint4 gid=gridIdx(mesh);
	if(inside(gid, mesh)){
		float u=(M_PI*(2*gid.x+1))/mesh.x;
		float v=(float)(2*gid.y)/(mesh.y-1)-1.f;
		float rho=scale*horner(d_Pml, l-abs(m), v)*powf(1.f-v*v, abs(m)/2)*cosf(m*u);
		d_rho[gid.w]+=rho*rho;
	}
}

__global__ void torus(float4 *d_vertex, dim3 mesh, float c, float a){
	uint4 gid=gridIdx(mesh);
	if(inside(gid, mesh)){
		float u=(2*gid.x*M_PI)/mesh.x;
		float v=(2*gid.y*M_PI)/mesh.y;
		cylindrical(d_vertex[gid.w], c+a*cosf(v), u, a*sinf(v));
	}
}
__global__ void mobius(float4 *d_vertex, dim3 mesh, float r, float w){
	uint4 gid=gridIdx(mesh);
	if(inside(gid, mesh)){
		float t=(2*gid.x*M_PI)/mesh.x;
		float s=w*((2*gid.y)/(mesh.y-1.f)-1.f);
		cylindrical(d_vertex[gid.w], r+s*cosf(t/2.f), t, s*sinf(t/2.f));
	}
}
__global__ void figureEight(float4 *d_vertex, dim3 mesh, float r, float w){
	uint4 gid=gridIdx(mesh);
	if(inside(gid, mesh)){
		float u=(2*gid.x*M_PI)/mesh.x;
		float v=(2*gid.y*M_PI)/mesh.y;
		float a=cosf(u/2);
		float b=sinf(u/2);
		float c=sinf(v);
		float d=sinf(2*v);
		cylindrical(d_vertex[gid.w], r*(w+a*c-b*d), u, r*(a*d+b*c));
	}
}
__global__ void catenoid(float4 *d_vertex, dim3 mesh, float c, float h){
	uint4 gid=gridIdx(mesh);
	if(inside(gid, mesh)){
		float u=((2*gid.x+1)*M_PI)/mesh.x;
		float z=h*(1.f-(2*gid.y)/(mesh.y-1.f));
		cylindrical(d_vertex[gid.w], c*coshf(z/c), u, z);
	}
}
__global__ void paraboloid(float4 *d_vertex, dim3 mesh, float c, float h){
	uint4 gid=gridIdx(mesh);
	if(inside(gid, mesh)){
		float u=((2*gid.x+1)*M_PI)/mesh.x;
		float r=h*(gid.y/(mesh.y-1.f));
		cylindrical(d_vertex[gid.w], r, u, r*r);
	}
}
__global__ void pretzel(float4 *d_vertex, dim3 mesh, float w){
	uint4 gid=gridIdx(mesh);
	if(inside(gid, mesh)){
		float u=(2*gid.y*M_PI)/mesh.y;
		float v=(2*gid.x*M_PI)/mesh.x;
		float p=1.25f+sinf(3*u);
		cylindrical(d_vertex[gid.w], w*(6.f-p*sinf(u-3*v)), v, w*p*cosf(u-3*v));
	}
}
__global__ void torusKnot(float4 *d_vertex, dim3 mesh, float r, float p, float q){
	uint4 gid=gridIdx(mesh);
	if(inside(gid, mesh)){
		float u=(2*gid.x*M_PI)/mesh.x;
		cylindrical(d_vertex[gid.w], r*(cosf(q*u)+2.f), p*u, -sinf(q*u));
	}
}
__global__ void dampedWave(float4 *d_vertex, dim3 mesh, float r, float t){
	uint4 gid=gridIdx(mesh);
	if(inside(gid, mesh)){
		float u=(2*gid.x*M_PI)/mesh.x;
		float v=(r*gid.y)/(mesh.y-1);
		float A=expf(-v*v/(r*r))*cosf(2*M_PI/(0.75*r)*v)*sinf(2*M_PI*t);
		cylindrical(d_vertex[gid.w], v, u, A);
	}
}

