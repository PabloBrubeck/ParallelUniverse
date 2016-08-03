/*
 * spectral.h
 *
 *  Created on: Aug 2, 2016
 *      Author: pbrubeck
 */

#ifndef SPECTRAL_H_
#define SPECTRAL_H_


#include "kernel.h"
#include "cufft.h"

__global__ void evenSymmetric(int n, double *d_u){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i>0 && i<n){
		d_u[2*(n-1)-i]=d_u[i];
	}
}


__global__ void diffFilter(int order, int n, cufftDoubleComplex *d_uhat){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		int k=2*i<n?i:i-n;
		k=pow(k,order);
		switch(order%4){
		case 0: // 1
			d_uhat[i].x=k*d_uhat[i].x/n;
			d_uhat[i].y=k*d_uhat[i].y/n;
			break;
		case 1: // -i
			d_uhat[i].x=-k*d_uhat[i].y/n;
			d_uhat[i].y=k*d_uhat[i].x/n;
			break;
		case 2: // -1
			d_uhat[i].x=-k*d_uhat[i].x/n;
			d_uhat[i].y=-k*d_uhat[i].y/n;
			break;
		case 3: // i
			d_uhat[i].x=k*d_uhat[i].y/n;
			d_uhat[i].y=-k*d_uhat[i].x/n;
			break;
		}
	}
}

void fftD(cufftHandle fftPlan, cufftHandle ifftPlan, int length, int order, double *d_v, double *d_u, cufftDoubleComplex *d_uhat){
	cufftExecD2Z(fftPlan, d_u, d_uhat);
	diffFilter<<<grid(length), MAXTHREADS>>>(order, length, d_uhat);
	cufftExecZ2D(ifftPlan, d_uhat, d_v);
}

void chebfftD(int n, double *d_u){

}

void chebfttD2(int n, double *d_u){

}

__global__
void chebNodes(int n, double *d_x){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_x[i]=cospi(i/(n-1.0));
	}
}

__global__
void chebDelem(int n, double *d_D, double *d_x){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<n && j<n){
		int gid=j*n+i;
		if(i!=j){
			int ci=(i==0 || i==n-1)?2:1;
			int cj=(j==0 || j==n-1)?2:1;
			d_D[gid]=(ci*((i+j)&1?-1:1))/(cj*(d_x[i]-d_x[j]));
		}else if(j>0 && j<n-1){
			d_D[gid]=-d_x[j]/(2*(1-d_x[j]*d_x[j]));
		}else{
			d_D[gid]=(j==0?1:-1)*(2*(n-1)*(n-1)+1)/6.0;
		}
	}
}

void chebD(int n, double *d_D, double *d_x){
	chebNodes<<<grid(n), MAXTHREADS>>>(n, d_x);
	chebDelem<<<grid(n,n), MAXTHREADS>>>(n, d_D, d_x);
}


#endif /* SPECTRAL_H_ */
