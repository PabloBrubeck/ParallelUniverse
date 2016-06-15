/*
 * SpectralMethods.h
 *
 *  Created on: May 25, 2016
 *      Author: pbrubeck
 */

#ifndef SPECTRALMETHODS_H_
#define SPECTRALMETHODS_H_

#include "kernel.h"
#include "cufft.h"

__global__ void evenSymmetric(double *d_u, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i>0 && i<n){
		d_u[2*(n-1)-i]=d_u[i];
	}
}


__global__ void diffFilter(cufftDoubleComplex *d_uhat, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		int k=2*i<n?i:i-n;
		double re=-k*d_uhat[i].y;
		double im=k*d_uhat[i].x;
		d_uhat[i].x=re/n;
		d_uhat[i].y=im/n;
	}
}

void fftD(cufftHandle fftPlan, cufftHandle ifftPlan, double *d_u, cufftDoubleComplex *d_uhat, int length, int order){
	cufftExecD2Z(fftPlan, d_u, d_uhat);
	diffFilter<<<grid(length), MAXTHREADS>>>(d_uhat, length);
	cufftExecZ2D(ifftPlan, d_uhat, d_u);
}

void chebfftD(double *d_u, int length){

}

void chebfttD2(double *d_u, int length){

}

__global__
void chebNodes(double *d_x, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_x[i]=cospi(i/(n-1.0));
	}
}

__global__
void chebDelem(double *d_D, double *d_x, int n){
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

void chebD(double *d_D, double *d_x, int n){
	chebNodes<<<grid(n), MAXTHREADS>>>(d_x, n);
	chebDelem<<<grid(n,n), MAXTHREADS>>>(d_D, d_x, n);
}

#endif /* SPECTRALMETHODS_H_ */
