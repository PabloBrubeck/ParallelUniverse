#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuMatlab.h"
#include "WaveSolver.h"

using namespace std;

void waveExample(int N){
	double *u=new double[N];
	linspace(u, -pi, pi, N);
	auto f=[](double th)->double{return th>0?0:(1-cos(2*th))/2;};
	map(u, u, N, f);
	double *d_u;
	cudaMalloc((void**)&d_u, 2*N*sizeof(double));
	cudaMemset(d_u, 0.0, 2*N*sizeof(double));
	cudaMemcpy(d_u, u, N*sizeof(double), cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);
	WaveSolver wave(handle, N, d_u, 0.0);

	int frames=144*10;
	double dt=6.0/(N*N);
	for(int i=0; i<frames; i++){
		wave.solve(dt);
	}
	cudaMemcpy(u, d_u, N*sizeof(double), cudaMemcpyDeviceToHost);
	disp(u, N, 1, 1);
	cublasDestroy(handle);
}

void cufftExample(int N){
	double *u=new double[N];
	linspace(u, -pi, pi, N);
	auto f=[](double th)->double{return cos(th);};
	map(u, u, N, f);
	double *d_u;
	cudaMalloc((void**)&d_u, N*sizeof(double));
	cudaMemcpy(d_u, u, N*sizeof(double), cudaMemcpyHostToDevice);

	cufftHandle fftPlan, ifftPlan;
	cufftPlan1d(&fftPlan, N, CUFFT_D2Z, 1);
	cufftPlan1d(&ifftPlan, N, CUFFT_Z2D, 1);

	cufftDoubleComplex *d_uhat;
	cudaMalloc((void**)&d_uhat, N*sizeof(cufftDoubleComplex));

	fftD(fftPlan, ifftPlan, N, 1, d_u, d_u, d_uhat);
	cudaMemcpy(u, d_u, N*sizeof(double), cudaMemcpyDeviceToHost);
	disp(u, N, 1, 1);

	cufftDestroy(fftPlan);
	cufftDestroy(ifftPlan);
}

void mapExample(int N){
	double *x=new double[N];
	double *y=new double[N];
	linspace(x, 0, pi, N);
	int m=0, deg=5;
	double *a=new double[deg]; a[deg-1]=1;
	auto f1=[a,deg,m](double th)->double{return LegendreP(a, deg, m, cos(th));};
	map(y, x, N, f1);
	disp(y,N,1,1);
}

void poisson(double ua, double ub, int n){
	double *d_x, *d_D, *d_D2;
	cudaMalloc((void**)&d_x, n*sizeof(double));
	cudaMalloc((void**)&d_D, n*n*sizeof(double));
	cudaMalloc((void**)&d_D2, n*n*sizeof(double));
	chebD(n, d_D, d_x);

	// compute second derivative operator D2=D*D
	cublasHandle_t cublasH;
	cublasCreate(&cublasH);
	double alpha=1;
	double beta=0;
	cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_D, n, d_D, n, &beta, d_D2, n);

	// right hand side
	// TODO: map on device, heterogeneous lambda
	double *h_b=new double[n];
	auto f=[](double x)->double{return 1.0;};
	map(h_b, h_b, n, f);
	double *d_b;
	cudaMalloc((void**)&d_b, n*sizeof(double));
	cudaMemcpy(d_b, h_b, n*sizeof(double), cudaMemcpyHostToDevice);

	alpha=-ua;
	cublasDaxpy(cublasH, n, &alpha, d_D2+n*(n-1), 1, d_b, 1);
	alpha=-ub;
	cublasDaxpy(cublasH, n, &alpha, d_D2        , 1, d_b, 1);
	cudaMemcpy(d_b,     &ub, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b+n-1, &ua, sizeof(double), cudaMemcpyHostToDevice);

	// solve Poisson D2*u=f(x)
	cusolverDnHandle_t cusolverH;
	cusolverDnCreate(&cusolverH);
	int lwork;
	int m=n-2, lda=n, ldb=n, nrhs=1;
	cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_D2+n+1, lda, &lwork);

	int    *d_info; cudaMalloc((void**)&d_info, sizeof(int));
	int    *d_Ipiv; cudaMalloc((void**)&d_Ipiv, m*sizeof(int));
	double *d_Work; cudaMalloc((void**)&d_Work, lwork*sizeof(double));

	cusolverDnDgetrf(cusolverH, m, m, d_D2+n+1, lda, d_Work, d_Ipiv, d_info);
	cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, nrhs, d_D2+n+1, lda, d_Ipiv, d_b+1, ldb, d_info);

	// display u
	cudaMemcpy(h_b, d_b, n*sizeof(double), cudaMemcpyDeviceToHost);
	disp(h_b, n, 1, 1);
	free(h_b);

	// free auxiliary variables
	cudaFree(d_Work);
	cudaFree(d_Ipiv);
	cudaFree(d_info);

	// free device memory
	cudaFree(d_x);
	cudaFree(d_D);
	cudaFree(d_D2);
	cudaFree(d_b);

	// destroy library handles
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
}

int main(int argc, char **argv){
	waveExample(1<<8);
	return 0;
}
