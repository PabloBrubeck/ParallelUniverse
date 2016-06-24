/*
 * WaveSolver.h
 *
 *  Created on: Jun 20, 2016
 *      Author: pbrubeck
 */

#ifndef WAVESOLVER_H_
#define WAVESOLVER_H_

#include "kernel.h"

class WaveSolver : public RungeKutta{
private:
	int length;
	cufftHandle fftPlan, ifftPlan;
	cufftDoubleComplex *u_hat;
public:
	WaveSolver(cublasHandle_t h, int n, double *u0, double t0) : RungeKutta(h,2*n,u0,t0){
		length=n;
		cufftPlan1d(&fftPlan, n, CUFFT_D2Z, 1);
		cufftPlan1d(&ifftPlan, n, CUFFT_Z2D, 1);
		cudaMalloc((void**)&u_hat, n*sizeof(cufftDoubleComplex));
    }
    void partialD(int n, double *dy, double *y, double ti);
};


void WaveSolver::partialD(int n, double *dy, double *y, double ti){
	fftD(fftPlan, ifftPlan, length, 1, dy, y+length, u_hat);
	fftD(fftPlan, ifftPlan, length, 1, dy+length, y, u_hat);
}



#endif /* WAVESOLVER_H_ */
