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
#include "linalg.h"
#include "integration.h"
#include "complex_functions.h"
#include "special_functions.h"
#include "spectral.h"
#include "RungeKutta.h"
#include "image.h"

using namespace std;

#define pi M_PI
#define eps DBL_EPSILON

/*
 * Helper functions (host)
 */

void disp(int m, double* A){
	for(int i=0; i<m; i++){
		printf("% .6e\n", A[i]);
	}
	printf("\n");
}

void disp(int m, int n, double* A, int lda){
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

template<typename F>
void mapHost(F fun, int n, double* x, double* y){
	for(int i=0; i<n; i++) {
	  y[i]=fun(x[i]);
	}
}

/*
 * CUDA helper functions (device)
 */

template<typename F, typename T1, typename T2>
__global__ void apply(F fun, int n, T1 *x, T2 *y){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		y[i]=fun(x[i]);
	}
}

template<typename F, typename T1, typename T2>
void cudaMap(F fun, int n, T1* x, T2* y){
	apply<<<grid(n), MAXTHREADS>>>(fun, n, x, y);
}

template<typename F, typename T1, typename T2>
__global__ void apply(F fun, int m, int n, T1* X, int ldx, T2* Y, int ldy){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<m && j<n){
		Y[j*ldy+i]=fun(X[j*ldx+i]);
	}
}

template<typename F, typename T1, typename T2>
void cudaMap(F fun, int m, int n, T1* X, int ldx, T2* Y, int ldy){
	apply<<<grid(m,n), MAXTHREADS>>>(fun, m, n, X, ldx, Y, ldy);
}

template<typename F, typename T1, typename T2>
__global__ void apply(F fun, int n, T1* y, T2 xmin, T2 xmax){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		T2 x=xmin+i*(xmax-xmin)/(n-1);
		y[i]=fun(x);
	}
}

template<typename F,typename T1, typename T2>
void cudaMap(F fun, int n, T1* y, T2 xmin, T2 xmax){
	apply<<<grid(n), MAXTHREADS>>>(fun, n, y, xmin, xmax);
}

template<typename F, typename T1, typename T2>
__global__ void apply(F fun, int m, int n, T1* A, int lda, T2 xmin, T2 xmax, T2 ymin, T2 ymax){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<m && j<n){
		T2 x=xmin+i*(xmax-xmin)/(m-1);
		T2 y=ymin+j*(ymax-ymin)/(n-1);
		A[j*lda+i]=fun(x,y);
	}
}

template<typename F,typename T1, typename T2>
void cudaMap(F fun, int m, int n, T1* A, int lda, T2 xmin, T2 xmax, T2 ymin, T2 ymax){
	apply<<<grid(m,n), MAXTHREADS>>>(fun, m, n, A, lda, xmin, xmax, ymin, ymax);
}

/*
 * Thrust helper functions (host & device)
 */

template<typename T> void zeros(int n, T* x){
	thrust::fill(x, x+n, (T)0);
}

template<typename T> void ones(int n, T* x){
	thrust::fill(x, x+n, (T)1);
}

template<typename T> void fill(T val, int n, T* x){
	thrust::fill(x, x+n, val);
}

template<typename T> void sequence(int n, T* x){
	thrust::sequence(x, x+n);
}

template<typename T> void linspace(T a, T b, int n, T* x){
	thrust::sequence(x, x+n, a, (b-a)/(n-1));
}

template<typename T> T sum(int n, T* x){
	return thrust::reduce(x, x+n, 0, thrust::plus<T>());
}

template<typename T> T prod(int n, T* x){
	return thrust::reduce(x, x+n, 1, thrust::multiplies<T>());
}

template<typename T> T mean(int n, T* x){
	return thrust::reduce(x, x+n, 0, thrust::plus<T>())/n;
}

template<typename T> T max(int n, T* x){
	return thrust::reduce(x, x+n, numeric_limits<T>::min(), thrust::maximum<T>());
}

template<typename T> T min(int n, T* x){
	return thrust::reduce(x, x+n, numeric_limits<T>::max(), thrust::minimum<T>());
}

template<typename T> void minmax(T *minptr, T *maxptr, int n, T* x){
	*minptr=thrust::reduce(x, x+n, numeric_limits<T>::max(), thrust::minimum<T>());
	*maxptr=thrust::reduce(x, x+n, numeric_limits<T>::min(), thrust::maximum<T>());
}


#endif /* CUMATLAB_H_ */
