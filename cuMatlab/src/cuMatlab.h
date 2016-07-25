/*
 * cuMatlab.h
 *
 *  Created on: May 23, 2016
 *      Author: pbrubeck
 */

#ifndef CUMATLAB_H_
#define CUMATLAB_H_

#include <math.h>
#include <functional>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "kernel.h"
#include "LinearAlgebra.h"
#include "Integration.h"
#include "SpecialFunctions.h"
#include "SpectralMethods.h"
#include "RungeKutta.h"

using namespace std;

#define pi M_PI
#define eps DBL_EPSILON

/*
 * Helper functions (host)
 */

void disp(double* A, int m, int n, int lda){
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			printf("% .6e\t", A[j*lda+i]);
		}
		printf("\n");
	}
	printf("\n");
}

void linspaceHost(double a, double b, int n, double* x){
	double h=(b-a)/(n-1);
	for(int i=0; i<n; i++){
		x[i]=a+h*i;
	}
}

void map(function<double(double)> fun, int n, double* y, double* x){
  for(int i=0; i<n; i++) {
	  y[i]=fun(x[i]);
  }
}

/*
 * CUDA helper functions (device)
 */

template<typename F, typename T> __global__ void apply(F fun, int n, T *d_x, T *d_y){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_y[i]=fun(d_x[i]);
	}
}

template<typename F, typename T> void cudaMap(F fun, int n, T* d_x, T* d_y){
	apply<<<grid(n), MAXTHREADS>>>(fun, n, d_x, d_y);
}


/*
 * Thrust helper functions (device)
 */

template<typename T> void zeros(int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	thrust::fill(t_x, t_x+n, (T)0);
}

template<typename T> void ones(int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	thrust::fill(t_x, t_x+n, (T)1);
}

template<typename T> void fill(T val, int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	thrust::fill(t_x, t_x+n, val);
}

template<typename T> void sequence(int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	thrust::sequence(t_x, t_x+n);
}

template<typename T> void linspace(T a, T b, int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	thrust::sequence(t_x, t_x+n, a, (b-a)/(n-1));
}

template<typename T> T sum(int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	return thrust::reduce(t_x, t_x+n, 0, thrust::plus<T>());
}

template<typename T> T prod(int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	return thrust::reduce(t_x, t_x+n, 1, thrust::multiplies<T>());
}

template<typename T> T mean(int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	return thrust::reduce(t_x, t_x+n, 0, thrust::plus<T>())/n;
}

template<typename T> T max(int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	return thrust::reduce(t_x, t_x+n, numeric_limits<T>::min(), thrust::maximum<T>());
}

template<typename T> T min(int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	return thrust::reduce(t_x, t_x+n, numeric_limits<T>::max(), thrust::minimum<T>());
}

template<typename T> void minmax(T *xmin, T *xmax, int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	*xmin=thrust::reduce(t_x, t_x+n, numeric_limits<T>::max(), thrust::minimum<T>());
	*xmax=thrust::reduce(t_x, t_x+n, numeric_limits<T>::min(), thrust::maximum<T>());
}




#endif /* CUMATLAB_H_ */
