/*
 * particle.h
 *
 *  Created on: Oct 18, 2016
 *      Author: pbrubeck
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "geometry.h"

class particle{
public:
	cublasHandle_t handle;
	int nbody, p;
	dim3 grid, block;
	float t, fr, L, radius, dL, EPS;
	float3 *x, *v, *f;
	float *f2;

	particle(cublasHandle_t h, int n, float3 *x0, float3 *v0, float fr);
    void verlet();
};

__global__ void crystal(int n, float3 *p, float L){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		p[i].x=L*fmodf(i*powf(n,-3.f/3.f), 1.f)-L/2;
		p[i].y=L*fmodf(i*powf(n,-2.f/3.f), 1.f)-L/2;
		p[i].z=L*fmodf(i*powf(n,-1.f/3.f), 1.f)-L/2;
	}
}


__device__ float3 forceLJ(float3 xi, float3 xj, float EPS, float L){
	float3 r, rij=xi-xj;
	float3 fij=make_float3(0,0,0);
	for(int n1=-1; n1<=1; n1++){
		r.x=rij.x+n1*L;
		for(int n2=-1; n2<=1; n2++){
			r.y=rij.y+n2*L;
			for(int n3=-1; n3<=1; n3++){
				r.z=rij.z+n3*L;
				float r2=dot(r,r);
				if(r2<L*L/4){
					float r6=r2*r2*r2;
					float r12=r6*r6;
					fij=4*EPS*(12/r12-6/r6)/r2*r;
					//Uij=4*EPS*(1/r12-1/r6);
				}
			}
		}
	}
	return fij;
}

__global__ void computeForce(int n, float3 *x, float3 *f, float EPS, float L){
	float3 *shared = SharedMemory<float3>();
	int src=blockIdx.x*blockDim.x+threadIdx.x;
	int dst=blockIdx.y*blockDim.x+threadIdx.x;
	if(src<n){
		shared[threadIdx.x]=x[src];
	}
	__syncthreads();
	if(dst<n){
		float3 xi=x[dst];
		float3 fi=make_float3(0.f, 0.f, 0.f);
		#pragma unroll 128
		for(int k=0; k<blockDim.x; k++){
			int j=blockIdx.x*blockDim.x+k;
			if(j<n && j!=dst){
				fi+=forceLJ(xi,shared[k],EPS,L);
			}
		}
		atomicAdd(&(f[dst].x), fi.x);
		atomicAdd(&(f[dst].y), fi.y);
		atomicAdd(&(f[dst].z), fi.z);
	}

}

void particle::verlet(){
	float delta=0;
	float L0=L;
	while(delta<1/fr){
		cudaMap([=] __device__ (float3 f){return dot(f,f);}, nbody, f, f2);

		cudaThreadSynchronize();
		float dt=min(1/fr, sqrt(dL/sqrt(max(nbody, f2))));

		float alpha=0.5*dt;
		cublasSaxpy(handle, 3*nbody, &alpha, (float*)f, 1, (float*)v, 1);
		cublasSaxpy(handle, 3*nbody, &dt,    (float*)v, 1, (float*)x, 1);

		cudaMap([L0] __device__ (float t){return fmodf(t+1.5f*L0, L0)-0.5f*L0;}, 3*nbody, (float*)x, (float*)x);
		cudaMemset(f, 0, nbody*sizeof(float3));
		computeForce<<<grid, block, p*sizeof(float3)>>>(nbody, x, f, EPS, L);

		cublasSaxpy(handle, 3*nbody, &alpha, (float*)f, 1, (float*)v, 1);
		delta+=dt;
	}
	t+=delta;
}

particle::particle(cublasHandle_t h, int n, float3 *x0,  float3 *v0, float framerate)
: handle(h), nbody(n), x(x0), v(v0), t(0), fr(framerate){
	p=1<<5;
	block=dim3(p,1,1);
	grid=dim3(ceil(n,p), ceil(n,p), 1);

	L=4.f*powf(nbody, 1.f/3.f);
	radius=powf(2.f,-5.f/6.f);
	dL=radius/1e5;
	EPS=10;

	cudaMalloc((void**)&f, nbody*sizeof(float3));
	cudaMallocManaged((void**)&f2, nbody*sizeof(float));

	crystal<<<(nbody+511)/512, 512>>>(nbody, x, L);
	cudaMemset(v, 0, nbody*sizeof(float3));
	cudaMemset(f, 0, nbody*sizeof(float3));
	computeForce<<<grid, block, p*sizeof(float3)>>>(nbody, x, f, EPS, L);
}

void particleShadder(dim3 mesh, float4* vertex, float4* norm, uchar4* color, uint4* index){
	static cublasHandle_t cublasH;
	static float3 *x=NULL, *v=NULL;
	static dim3 grid3, block3;
	static float fr=200;

	if(x==NULL){
		cublasCreate(&cublasH);
		gridblock(grid3, block3, mesh);
		cudaMalloc((void**)&x, mesh.z*sizeof(float3));
		cudaMalloc((void**)&v, mesh.z*sizeof(float3));
		indexS2<<<grid3, block3>>>(mesh, index);
		colorSpheres<<<grid3, block3>>>(mesh, color);
	}

	{
		static particle P(cublasH, mesh.z, x, v, fr);
		P.verlet();
	}
	spheres<<<grid3, block3>>>(mesh, x, vertex, norm, powf(2.f, -5.f/6.f));
}

#endif /* PARTICLE_H_ */
