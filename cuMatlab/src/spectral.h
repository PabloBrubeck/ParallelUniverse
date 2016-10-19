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

#include "strided_range.h"
#include <thrust/scan.h>

__global__ void timesi(int N, cuDoubleComplex *z){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N){
		z[i]=make_cuDoubleComplex(-z[i].y, z[i].x);
	}
}

__global__ void auxdst(int N, double *f, double *y){
	int j=blockIdx.x*blockDim.x+threadIdx.x;
	if(j<N){
		int Nmj=(N-j)%N;
		y[j]=sinpi((double)j/N)*(f[j]+f[Nmj])+0.5*(f[j]-f[Nmj]);
	}
}

void dst(int N, double *f, double *F){
	auxdst<<<grid(N),MAXTHREADS>>>(N, f, F);

	cufftHandle fftPlan;
	cufftPlan1d(&fftPlan, N, CUFFT_D2Z, 1);
	cufftExecD2Z(fftPlan, F, (cufftDoubleComplex*)F);
	timesi<<<grid(N/2),MAXTHREADS>>>(N/2, (cufftDoubleComplex*)F);

	double F1;
	cudaMemcpy(&F1, F+1, sizeof(double), cudaMemcpyDeviceToHost);
	F1*=0.5;
	cudaMemcpy(F+1, &F1, sizeof(double), cudaMemcpyHostToDevice);

	thrust::device_ptr<double> ptr(F);
	strided_range<thrust::device_vector<double>::iterator>  odds(ptr+1, ptr+N, 2);
	thrust::inclusive_scan(odds.begin(), odds.end(), odds.begin());
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
		if(i!=j){
			int ci=(i==0 || i==n-1)?2:1;
			int cj=(j==0 || j==n-1)?2:1;
			D[j*n+i]=((i+j)&1?-ci:ci)/(cj*(x[i]-x[j]));
		}else if(j>0 && j<n-1){
			D[j*n+i]=-x[j]/(2*(1-x[j]*x[j]));
		}else{
			D[j*n+i]=(j==0?1:-1)*(2*(n-1)*(n-1)+1)/6.0;
		}
	}
}

void chebD(int n, double *D, double *x){
	chebNodes<<<grid(n), MAXTHREADS>>>(n, x);
	chebDelem<<<grid(n,n), MAXTHREADS>>>(n, D, x);
}


#endif /* SPECTRAL_H_ */
