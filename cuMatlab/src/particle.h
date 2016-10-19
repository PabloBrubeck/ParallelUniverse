/*
 * particle.h
 *
 *  Created on: Oct 18, 2016
 *      Author: pbrubeck
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

__global__ void lennardjones(int n, float3 *x, float3 *force, float EPS){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<n && j<n && i!=j){
		float3 r=x[i]-x[j];
		float r2=dot(r,r);
		float r6=r2*r2*r2;
		float r12=r6*r6;
		force[i]+=4*EPS*(12/r12-6/r6)/r2*r;
	}
}

__global__ void verlet(int n, float3 *x1, float3 *x0, float3 *force, float dt){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		x1[i]=2*x1[i]-x0[i]+dt*dt*force[i];
	}
}


#endif /* PARTICLE_H_ */
