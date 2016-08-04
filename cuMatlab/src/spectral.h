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


__global__ void diffFilter(double order, int n, cufftDoubleComplex *uhat){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	if(gid<n){
		int k=(2*gid<n)? gid : gid-n;
		if(order==1){
			uhat[gid].x=-k*uhat[gid].y/n;
			uhat[gid].y= k*uhat[gid].x/n;
		}else{
			cuDoubleComplex ik=make_cuDoubleComplex(0, k);
			uhat[gid]=pow(ik, order)*uhat[gid]/n;
		}
	}
}

void fftD(cufftHandle fftPlan, cufftHandle ifftPlan, int length, double order, double *v, double *u, cufftDoubleComplex *uhat){
	cufftExecD2Z(fftPlan, u, uhat);
	diffFilter<<<grid(length), MAXTHREADS>>>(order, length, uhat);
	cufftExecZ2D(ifftPlan, uhat, v);
}

void chebfftD(int n, double *u){

}

void chebfttD2(int n, double *u){

}

__global__
void chebNodes(int n, double *x){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		x[i]=cospi(i/(n-1.0));
	}
}

__global__
void chebDelem(int n, double *D, double *x){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<n && j<n){
		int gid=j*n+i;
		if(i!=j){
			int ci=(i==0 || i==n-1)?2:1;
			int cj=(j==0 || j==n-1)?2:1;
			D[gid]=(ci*((i+j)&1?-1:1))/(cj*(x[i]-x[j]));
		}else if(j>0 && j<n-1){
			D[gid]=-x[j]/(2*(1-x[j]*x[j]));
		}else{
			D[gid]=(j==0?1:-1)*(2*(n-1)*(n-1)+1)/6.0;
		}
	}
}

void chebD(int n, double *D, double *x){
	chebNodes<<<grid(n), MAXTHREADS>>>(n, x);
	chebDelem<<<grid(n,n), MAXTHREADS>>>(n, D, x);
}


#endif /* SPECTRAL_H_ */
