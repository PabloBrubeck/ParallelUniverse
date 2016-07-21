/*
 * cuMatlab.h
 *
 *  Created on: May 23, 2016
 *      Author: pbrubeck
 */

#ifndef CUMATLAB_H_
#define CUMATLAB_H_

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <functional>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "kernel.h"
#include "LinearAlgebra.h"
#include "Integration.h"
#include "SpecialFunctions.h"
#include "SpectralMethods.h"
#include "RungeKutta.h"

using namespace std;
using namespace thrust;

#define pi M_PI
#define eps DBL_EPSILON

void disp(double* A, int m, int n, int lda){
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			printf("% .6e\t", A[j*lda+i]);
		}
		printf("\n");
	}
	printf("\n");
}

void linspace(double* x, double a, double b, int n){
	double h=(b-a)/(n-1);
	for(int i=0; i<n; i++){
		x[i]=a+h*i;
	}
}

void map(double* y, double* x, int length, function<double(double)> fun){
  for(int i=0; i<length; i++) {
	  y[i]=fun(x[i]);
  }
}

void thrustMap(int n, double* d_x, double* d_y, function<double(double)> fun){
	device_ptr<double> t_x(d_x);
	device_ptr<double> t_y(d_y);
	//transform(t_x, t_x+n, t_y, fun);
}

template<typename T> T thrustMax(int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	return thrust::reduce(t_x, t_x+n, -FLT_MAX, thrust::maximum<T>());
}

template<typename T> T thrustMin(int n, T* d_x){
	thrust::device_ptr<T> t_x(d_x);
	return thrust::reduce(t_x, t_x+n,  FLT_MAX, thrust::minimum<T>());
}


#endif /* CUMATLAB_H_ */
