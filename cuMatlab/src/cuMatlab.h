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
#include "image.h"

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

template<typename F, typename T1, typename T2>
__global__ void apply(F fun, int n, T1 *d_x, T2 *d_y){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_y[i]=fun(d_x[i]);
	}
}

template<typename F, typename T1, typename T2>
void cudaMap(F fun, int n, T1* d_x, T2* d_y){
	apply<<<grid(n), MAXTHREADS>>>(fun, n, d_x, d_y);
}

template<typename F, typename T1, typename T2>
__global__ void apply(F fun, int m, int n, T1* d_x, int ldx, T2* d_y, int ldy){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<m && j<n){
		d_y[j*ldy+i]=fun(d_x[i*ldx+j]);
	}
}

template<typename F, typename T1, typename T2>
void cudaMap(F fun, int m, int n, T1* d_x, int ldx, T2* d_y, int ldy){
	apply<<<grid(m,n), MAXTHREADS>>>(fun, m, n, d_x, ldx, d_y, ldy);
}

template<typename F, typename T1, typename T2>
__global__ void apply(F fun, int n, T1* d_A, T2 xmin, T2 xmax){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		T2 x=xmin+i*(xmax-xmin)/(n-1);
		d_A[i]=fun(x);
	}
}

template<typename F,typename T1, typename T2>
void cudaMap(F fun, int n, T1* d_A, T2 xmin, T2 xmax){
	apply<<<grid(n), MAXTHREADS>>>(fun, n, d_A, xmin, xmax);
}

template<typename F, typename T1, typename T2>
__global__ void apply(F fun, int m, int n, T1* d_A, int lda, T2 xmin, T2 xmax, T2 ymin, T2 ymax){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<m && j<n){
		T2 x=xmin+i*(xmax-xmin)/(m-1);
		T2 y=ymin+j*(ymax-ymin)/(n-1);
		d_A[j*lda+i]=fun(x,y);
	}
}

template<typename F,typename T1, typename T2>
void cudaMap(F fun, int m, int n, T1* d_A, int lda, T2 xmin, T2 xmax, T2 ymin, T2 ymax){
	apply<<<grid(m,n), MAXTHREADS>>>(fun, m, n, d_A, lda, xmin, xmax, ymin, ymax);
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

template<typename T> void minmax(T *minptr, T *maxptr, int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	*minptr=thrust::reduce(t_x, t_x+n, numeric_limits<T>::max(), thrust::minimum<T>());
	*maxptr=thrust::reduce(t_x, t_x+n, numeric_limits<T>::min(), thrust::maximum<T>());
}


#endif /* CUMATLAB_H_ */
