/*
 * Integration.h
 *
 *  Created on: May 25, 2016
 *      Author: pbrubeck
 */

#ifndef INTEGRATION_H_
#define INTEGRATION_H_

#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__

#include "LinearAlgebra.h"

// Gaussian quadratures using the Golub-Welsch algorithm

void gaucheb(int n, double *x, double *w, double a, double b){
	double x0=(b+a)/2;
	double dx=(b-a)/2;
	double wi=(b-a)*M_PI/(2*n);
	for(int i=0; i<n; i++){
		x[i]=x0+dx*cospi((2*i+1.0)/(2*n));
		w[i]=wi;
	}
}

void gauleg(int n, double *x, double *w, double a, double b){
	double *D=new double[n];
	double *E=new double[n];
	double *W=new double[n*n];
	for(int i=0; i<n; i++){
		int k=i+1;
		D[i]=0;
		E[i]=k/sqrt(4*k*k-1);
	}

	trideig(n, W, x, D, E);
	double x0=(b+a)/2;
	double dx=(b-a)/2;
	double dw=b-a;
	for(int i=0; i<n; i++){
		x[i]=x0+dx*x[i];
		w[i]=dw*W[i*n]*W[i*n];
	}
}

void gaulag(int n, double *x, double *w){
	double *D=new double[n];
	double *E=new double[n];
	double *W=new double[n*n];
	for(int i=0; i<n; i++){
		int k=i+1;
		D[i]=2*k-1;
		E[i]=k;
	}
	trideig(n, W, x, D, E);
	for(int i=0; i<n; i++){
		w[i]=W[i*n]*W[i*n];
	}
}

void gauherm(int n, double *x, double *w, double mu, double sigma){
	double *D=new double[n];
	double *E=new double[n];
	double *W=new double[n*n];
	for(int i=0; i<n; i++){
		D[i]=0;
		E[i]=sqrt((i+1)/2.0);
	}
	trideig(n, W, x, D, E);
	double dw=sqrt(2*M_PI);
	double dx=M_SQRT2*sigma;
	for(int i=0; i<n; i++){
		x[i]=mu+dx*x[i];
		w[i]=dw*W[i*n]*W[i*n];
	}
}

#endif /* INTEGRATION_H_ */
