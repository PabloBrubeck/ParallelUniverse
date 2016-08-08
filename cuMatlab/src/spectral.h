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
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/scan.h>

__global__ void auxdst(int N, double *y, double *f){
	int j=blockIdx.x*blockDim.x+threadIdx.x;
	if(j<N){
		int Nmj=(N-j)%N;
		y[j]=sinpi((double)j/N)*(f[j]+f[Nmj])+0.5*(f[j]-f[Nmj]);
	}
}

void dst(int N, double *f, double *F){
	double *y;
	cudaMalloc((void**)&y, N*sizeof(double));
	auxdst<<<grid(N),MAXTHREADS>>>(N, y, f);

	cufftHandle fftPlan;
	cufftPlan1d(&fftPlan, N, CUFFT_D2Z, 1);
	cufftExecD2Z(fftPlan, y, (cufftDoubleComplex*)F);

	thrust::inclusive_scan(F, F+N, F);
}


__global__ void diffFilter(double order, int n, cufftDoubleComplex *uhat){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	if(gid<n){
		int k=(2*gid<n)? gid : gid-n;
		if(order==1){
			uhat[gid]=make_cuDoubleComplex(-k*uhat[gid].y/n, k*uhat[gid].x/n);
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
		x[i]=cospi(i/(double)(n-1));
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
			D[gid]=((i+j)&1?-ci:ci)/(cj*(x[i]-x[j]));
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
