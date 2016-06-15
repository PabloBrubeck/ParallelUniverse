/*
 * RungeKutta.h
 *
 *  Created on: Jun 13, 2016
 *      Author: pbrubeck
 */

#ifndef RUNGEKUTTA_H_
#define RUNGEKUTTA_H_

#include "kernel.h"

class RungeKutta{
	double *u, *y, t;
	double *k1, *k2, *k3, *k4;
	int length;
	cublasHandle_t handle;
	public:
    RungeKutta(cublasHandle_t h, int n, double *u0, double t0){
    	handle=h;
    	length=n;
    	u=u0;
    	t=t0;
    	cudaMalloc((void**)&y, n*sizeof(double));
    	cudaMalloc((void**)&k1, n*sizeof(double));
    	cudaMalloc((void**)&k2, n*sizeof(double));
    	cudaMalloc((void**)&k3, n*sizeof(double));
    	cudaMalloc((void**)&k4, n*sizeof(double));
    }
    void partialD(int n, double *dy, double *y, double ti);
    void step(double *ka, double *kb, double alpha);
	void solve(double dt);
};

__global__
void partialDKernel(int n, double *dy, double *y, double ti){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		dy[i]=y[i];
	}
}

void RungeKutta::partialD(int n, double *dy, double *y, double ti){
	partialDKernel<<<grid(n), MAXTHREADS>>>(n, dy, y, ti);
}

void RungeKutta::step(double *ka, double *kb, double alpha){
	double ti=t+alpha;
	cublasDcopy(handle, length, u, 1, y, 1);
	cublasDaxpy(handle, length, &alpha, ka, 1, y, 1);
	partialD(length, kb, y, ti);
}

void RungeKutta::solve(double dt){
	step(k4, k1, 0);
	step(k1, k2, dt/2);
	step(k2, k3, dt/2);
	step(k3, k4, dt);

	double alpha=2;
	cublasDaxpy(handle, length, &alpha, k2, 1, k1, 1);
	cublasDaxpy(handle, length, &alpha, k3, 1, k1, 1);
	alpha=1;
	cublasDaxpy(handle, length, &alpha, k4, 1, k1, 1);
	alpha=dt/6;
	cublasDaxpy(handle, length, &alpha, k1, 1, u, 1);
	t+=dt;
}


#endif /* RUNGEKUTTA_H_ */
