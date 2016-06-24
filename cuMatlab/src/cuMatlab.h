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

#include "kernel.h"
#include "LinearAlgebra.h"
#include "Integration.h"
#include "SpecialFunctions.h"
#include "SpectralMethods.h"
#include "RungeKutta.h"


using namespace std;

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


#endif /* CUMATLAB_H_ */
