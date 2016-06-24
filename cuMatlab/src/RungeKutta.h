/*
 * RungeKutta.h
 *
 *  Created on: Jun 13, 2016
 *      Author: pbrubeck
 */

#ifndef RUNGEKUTTA_H_
#define RUNGEKUTTA_H_

class RungeKutta{
private:
	int length;
public:
	double *u, *y, t;
	double *k1, *k2, *k3, *k4;
	cublasHandle_t handle;
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
    virtual void partialD(int n, double *dy, double *y, double ti) = 0;
    void step(double *ka, double *kb, double alpha);
	void solve(double dt);
};

void RungeKutta::step(double *ka, double *kb, double alpha){
	cublasDcopy(handle, length, u, 1, y, 1);
	cublasDaxpy(handle, length, &alpha, ka, 1, y, 1);
	partialD(length, kb, y, t+alpha);
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
