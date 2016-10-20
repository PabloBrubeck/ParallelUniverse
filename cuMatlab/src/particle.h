/*
 * particle.h
 *
 *  Created on: Oct 18, 2016
 *      Author: pbrubeck
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

class particle{
public:
	cublasHandle_t handle;
	int nbody;
	dim3 grid, block;
	float t, fr, L, radius, dL, EPS;
	float3 *x, *v, *f;
	float *U, *f2;

	particle(cublasHandle_t h, int n, float3 *x0, float3 *v0, float fr);
    void verletvel();
};

__global__ void recenter(int n, float3 *p, float L){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		p[i].x=L*(p[i].x-0.5f);
		p[i].y=L*(p[i].y-0.5f);
		p[i].z=L*(p[i].z-0.5f);
	}
}

__global__ void lennardjones(int n, float3 *x, float3 *f, float *U, float EPS, float L){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<n && j<n && i!=j){
		float3 r, r0=x[i]-x[j];
		for(int n1=-1; n1<=1; n1++){
			r.x=r0.x+n1*L;
			for(int n2=-1; n2<=1; n2++){
				r.y=r0.y+n2*L;
				for(int n3=-1; n3<=1; n3++){
					r.z=r0.z+n3*L;
					float r2=dot(r,r);
					if(r2<L^2/4){
						float r6=r2*r2*r2;
						float r12=r6*r6;
						float Uij=4*EPS*(1/r12-1/r6);
						float3 fij=4*EPS*(12/r12-6/r6)/r2*r;
						atomicAdd(&(U[i]),   Uij);
						atomicAdd(&(f[i].x), fij.x);
						atomicAdd(&(f[i].y), fij.y);
						atomicAdd(&(f[i].z), fij.z);
					}
				}
			}
		}
	}
}

particle::particle(cublasHandle_t h, int n, float3 *x0,  float3 *v0, float framerate)
: handle(h), nbody(n), x(x0), v(v0), t(0), fr(framerate){
	gridblock(grid, block, dim3(n,n,1));
	L=4.f*powf(nbody, 1.f/3.f);
	radius=powf(2.f,-5.f/6.f);
	dL=radius/1e5;
	EPS=10;


	cudaMallocManaged((void**)&U, nbody*sizeof(float));
	cudaMallocManaged((void**)&f2, nbody*sizeof(float));
	cudaMallocManaged((void**)&f, nbody*sizeof(float3));


	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(generator, 1000000ULL);

	float Ut=0;
	while(Ut>=0){
		curandGenerateUniform(generator, (float*)x, 3*nbody);
		recenter<<<(nbody+511)/512, 512>>>(nbody, x, L);

		cudaMemset(U, 0, nbody*sizeof(float));
		cudaMemset(v, 0, nbody*sizeof(float3));
		cudaMemset(f, 0, nbody*sizeof(float3));

		lennardjones<<<grid, block>>>(nbody, x, f, U, EPS, L);
		cudaThreadSynchronize();
		Ut=sum(nbody, U)/2;
		printf("%f\n", Ut);
	}
}

void particle::verletvel(){
	float delta=0;
	float Lm=0.5*L;
	float L0=L;
	while(delta<1/fr){
		cudaMap([=] __device__ (float3 f){return dot(f,f);}, nbody, f, f2);

		cudaThreadSynchronize();
		float dt=min(1/fr, sqrt(dL/sqrt(max(nbody, f2))));

		float alpha=0.5*dt;
		cublasSaxpy(handle, 3*nbody, &alpha, (float*)f, 1, (float*)v, 1);
		cublasSaxpy(handle, 3*nbody, &dt,    (float*)v, 1, (float*)x, 1);

		cudaMap([L0,Lm] __device__ (float t){return fmodf(t+Lm+L0, L0)-Lm;}, 3*nbody, (float*)x, (float*)x);
		cudaMemset(f, 0, nbody*sizeof(float3));
		lennardjones<<<grid, block>>>(nbody, x, f, U, EPS, L);
		cublasSaxpy(handle, 3*nbody, &alpha, (float*)f, 1, (float*)v, 1);
		delta+=dt;
	}
	t+=delta;
}


void particleExample(dim3 mesh, float4* vertex, float4* norm, uchar4* color, uint4* index){
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
		colorSurf<<<grid3, block3>>>(mesh, color);
	}

	{
		static particle P(cublasH, mesh.z, x, v, fr);
		P.verletvel();
	}
	spheres<<<grid3, block3>>>(mesh, x, vertex, norm, powf(2.f, -5.f/6.f));
}

#endif /* PARTICLE_H_ */
