/*
 * geometry.h
 *
 *  Created on: Oct 18, 2016
 *      Author: pbrubeck
 */

#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


inline __host__ __device__
float4 cross(float4 &a, float4 &b){
	return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0.f);
}

inline __host__ __device__  void cartesian(float4 &p, float x, float y, float z){
	p.x=x;
	p.y=y;
	p.z=z;
	p.w=1.f;
}
inline __host__ __device__  void cylindrical(float4 &p, float s, float phi, float z){
	p.x=s*cosf(phi);
	p.y=s*sinf(phi);
	p.z=z;
	p.w=1.f;
}
inline __host__ __device__ void spherical(float4 &p, float r, float theta, float phi){
	float rho=r*sinf(theta);
	p.x=rho*cosf(phi);
	p.y=rho*sinf(phi);
	p.z=r*cosf(theta);
	p.w=1.f;
}

__global__ void normalMapping(dim3 mesh,  float4 *vertex, float4 *norm){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;

		int ii=(i+1)%mesh.x;
		int jj=(j+1)%mesh.y;

		float4 A=vertex[gid];
		float4 T=vertex[(k*mesh.y+j)*mesh.x+ii]-A;
		float4 B=vertex[(k*mesh.y+jj)*mesh.x+i]-A;
		float4 N=normalize(cross(T, B));
		norm[gid]=N;
	}
}

__global__ void normalMapping(dim3 mesh, float4 *vertex, float4 *norm, uchar4 *color){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;

		int ii=(i+1)%mesh.x;
		int jj=(j+1)%mesh.y;

		float4 A=vertex[gid];
		float4 T=vertex[(k*mesh.y+j)*mesh.x+ii]-A;
		float4 B=vertex[(k*mesh.y+jj)*mesh.x+i]-A;
		float4 N=normalize(cross(T, B));
		norm[gid]=N;

		float x=N.x,  y=N.y,  z=N.z;
		float x2=x*x, y2=y*y, z2=z*z;

		float r=(x<0?0:x2)+(y<0?y2:0)+(z<0?z2:0);
		float g=(x<0?x2:0)+(y<0?0:y2)+(z<0?z2:0);
		float b=(x<0?x2:0)+(y<0?y2:0)+(z<0?0:z2);

		color[gid].x=(unsigned char)(255.f*r)&0xff;
		color[gid].y=(unsigned char)(255.f*g)&0xff;
		color[gid].z=(unsigned char)(255.f*b)&0xff;
		color[gid].w=192u;
	}
}

__global__ void colorSurf(dim3 mesh, uchar4 *color){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		color[gid]=hsv2rgb((float)k/mesh.z, 1.f, 1.f);
		color[gid].w=192u;
	}
}


__global__ void circle(int n, float4 *pos, float R){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		float phi=2*i*M_PI/n;
		cylindrical(pos[i], R, phi, 0);
	}
}

__global__ void torus(dim3 mesh, float4 *vertex, float c, float a){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<mesh.x && j<mesh.y){
		float u=(2*i*M_PI)/mesh.x;
		float v=(2*j*M_PI)/mesh.y;
		cylindrical(vertex[i+mesh.x*j], c+a*cosf(v), u, a*sinf(v));
	}
}

__global__ void indexT2(dim3 mesh, uint4* index){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;

		int ii=(i+1)%mesh.x;
		int jj=(j+1)%mesh.y;
		index[gid].x=(k*mesh.y+j)*mesh.x+i;
		index[gid].y=(k*mesh.y+j)*mesh.x+ii;
		index[gid].z=(k*mesh.y+jj)*mesh.x+ii;
		index[gid].w=(k*mesh.y+jj)*mesh.x+i;
	}
}

__global__ void spheres(dim3 mesh, float4 *pos, float4 *vertex, float4 *norm, float R){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=i+mesh.x*(j+mesh.y*k);
		float u=(float)i/(mesh.x-1);
		float v=(2*j*M_PI)/mesh.y;

		float s=2.f*sqrtf(u*(1.f-u));
		float z=2.f*u-1.f;

		//spherical(norm[gid], 1.f, u, v);
		cylindrical(norm[gid], s, v, z);
		vertex[gid]=R*norm[gid]+pos[k];
		vertex[gid].w=1.f;
	}
}

__global__ void indexS2(dim3 mesh, uint4 *index){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=i+mesh.x*(j+mesh.y*k);
		int ii=min(i+1, mesh.x-1);
		int jj=(j+1)%mesh.y;
		index[gid].x=(k*mesh.y+j)*mesh.x+i;
		index[gid].y=(k*mesh.y+j)*mesh.x+ii;
		index[gid].z=(k*mesh.y+jj)*mesh.x+ii;
		index[gid].w=(k*mesh.y+jj)*mesh.x+i;
	}
}



#endif /* GEOMETRY_H_ */
