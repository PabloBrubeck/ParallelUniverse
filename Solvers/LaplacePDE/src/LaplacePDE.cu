/*
 ============================================================================
 Name        : LaplacePDE.cu
 Author      : Pablo Brubeck
 Version     : 1.0.0
 Copyright   : Copyright 2015. All rights reserved.
 Description : Solves Laplace's Equation subject to Dirichlet conditions.
 ============================================================================
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace std;

typedef float (*func)(float x);

__device__ func psinh = sinhf;

__global__ void linspace(float *d_array, float a, float b, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_array[i]=a+i*(b-a)/(n-1);
	}
}
__global__ void constMult(float c, float *d_A, float *d_B,  int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_B[i]=c*d_A[i];
	}
}
__global__ void arrayMult(float *d_A, float *d_B, float *d_C, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_C[i]=d_A[i]*d_B[i];
	}
}
__global__ void arrayDivide(float *d_A, float *d_B, float *d_C, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_C[i]=d_A[i]/d_B[i];
	}
}
__global__ void diagMult(float *d_diag, float *d_A, float *d_B, int m, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<m && j<n){
		d_B[i*n+j]=d_diag[i]*d_A[i*n+j];
	}
}
__global__ void outerp(float *d_A, float *d_B, float *d_C, int m, int r){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<m && j<r){
		d_C[i*r+j]=d_A[i]*d_B[j];
	}
}
__global__ void mmult(float *d_A, float *d_B, float *d_C, int m, int n, int r){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<m && j<r){
		float temp=0.f;
		for(int k=0; k<n; k++){
			temp+=d_A[i*n+k]*d_B[k*r+j];
		}
		d_C[i*r+j]=temp;
	}
}


__global__ void mapFunction(func f, float *d_in, float *d_out, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_out[i]=(*f)(d_in[i]);
	}
}

inline int ceil(int num, int den){
	return (num+den-1)/den;
}
inline int nextPow2(int x){
    if(x < 0){return 0;}
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

void disp(float* d_array, int n){
	float *h_temp=new float[n];
	cudaMemcpy(h_temp, d_array, n*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0; i<n; i++){
		cout<<h_temp[i]<<endl;
	}
	delete[] h_temp;
}


int main(void){
	int m=16;
	int n=1<<11;
	int threads=(n<512)? nextPow2(n): 512;

	dim3 Mgrid(ceil(m, threads));
	dim3 Ngrid(ceil(n, threads));
	dim3 MNblock(32,32);
	dim3 MNgrid(ceil(m, 32), ceil(n, 32));

	func h_f;
	float *d_c, *d_k, *d_y, *d_A, *d_u;
	cudaMalloc((void**) &d_c, m*sizeof(float));
	cudaMalloc((void**) &d_k, m*sizeof(float));
	cudaMalloc((void**) &d_y, n*sizeof(float));
	cudaMalloc((void**) &d_A, m*n*sizeof(float));
	cudaMalloc((void**) &d_u, n*n*sizeof(float));

	linspace<<<Ngrid, threads>>>(d_k, M_PI, m*M_PI, m);
	linspace<<<Ngrid, threads>>>(d_y, 0.f, 1.f, n);
	outerp<<<MNgrid, MNblock>>>(d_k, d_y, d_A, m, n);

	cudaMemcpyFromSymbol(&h_f, psinh, sizeof(func));
	mapFunction<<<Mgrid, threads>>>(h_f, d_k, d_k, m);
	mapFunction<<<ceil(m*n, threads), threads>>>(h_f, d_A, d_A, m*n);

	arrayDivide<<<Mgrid, threads>>>(d_c, d_k, d_c, m);
	diagMult<<<MNgrid, MNblock>>>(d_c, d_A, d_A, m, n);

	return 0;
}
