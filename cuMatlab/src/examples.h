/*
 * examples.h
 *
 *  Created on: Aug 1, 2016
 *      Author: pbrubeck
 */

#ifndef EXAMPLES_H_
#define EXAMPLES_H_

#include "WaveSolver.h"

void mapExample(int n){
	// Maps a function f(x) on the domain [0, pi]

	double *d_x;
	cudaMalloc((void**)&d_x, n*sizeof(double));
	linspace(0.0, pi, n, d_x);

	int len=4;
	double *h_a, *d_a;
	h_a=new double[len]; h_a[len-1]=1;
	cudaMalloc((void**)&d_a, n*sizeof(double));
	cudaMemcpy(d_a, h_a, len*sizeof(double), cudaMemcpyHostToDevice);

	int m=0;
	auto lambda = [len, d_a, m] __device__ (double t){
		return LegendreP(len,d_a,m,cos(t));
	};
	cudaMap(lambda, n, d_x, d_x);


	double *h_x=new double[n];
	cudaMemcpy(h_x, d_x, n*sizeof(double), cudaMemcpyDeviceToHost);
	disp(h_x, n, 1, 1);
}


void quadratureExample(int n){
	// Computes the integral of f(x) on [-1,1]

	double *x=new double[n];
	double *w=new double[n];
	gauleg(n,x,w,-1,1);

	auto f=[] (double x){
		return x*x;
	};

	map(f,n,x,x);
	int inc=1;
	double J=ddot(&n,x,&inc,w,&inc);
	printf("%.15f\n",J);
}

void poisson(double ua, double ub, int n){
	// Solves the Dirichlet problem u_xx = 1, u(a)=ua, u(b)=ub

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
	double *d_b;
	cudaMalloc((void**)&d_b, n*sizeof(double));
	ones(n, d_b);

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
	double *h_u=new double[n];
	cudaMemcpy(h_u, d_b, n*sizeof(double), cudaMemcpyDeviceToHost);
	disp(h_u, n, 1, 1);

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

void waveExample(int N){
	// Solves the one-dimensional wave equation u_xx = u_tt, given initial u(x) and du/dt

	double *h_u=new double[N];
	linspace(-pi, pi, N, h_u);
	auto f=[](double th){return th>0?0:(1-cos(2*th))/2;};
	map(f, N, h_u, h_u);
	double *d_u;
	cudaMalloc((void**)&d_u, 2*N*sizeof(double));
	cudaMemset(d_u, 0, 2*N*sizeof(double));

	cublasHandle_t handle;
	cublasCreate(&handle);
	WaveSolver wave(handle, N, d_u, 0.0);

	int frames=144*10;
	double dt=6.0/(N*N);
	for(int i=0; i<frames; i++){
		wave.solve(dt);
	}
	cudaMemcpy(h_u, d_u, N*sizeof(double), cudaMemcpyDeviceToHost);
	disp(h_u, N, 1, 1);
	cublasDestroy(handle);
}

void cufftExample(int N){
	// Computes first derivative of a periodic function f(x) on [-pi, pi]

	double *u=new double[N];
	linspace(-pi, pi, N, u);
	auto f=[](double th)->double{return cos(th);};
	map(f, N, u, u);
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



#endif /* EXAMPLES_H_ */
