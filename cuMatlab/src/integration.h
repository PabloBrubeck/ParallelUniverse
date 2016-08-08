/*
 * integration.h
 *
 *  Created on: Aug 2, 2016
 *      Author: pbrubeck
 */

#ifndef INTEGRATION_H_
#define INTEGRATION_H_


#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__

#include "linalg.h"

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
	for(int k=1; k<=n; k++){
		D[k-1]=0;
		E[k-1]=k/sqrt(4*k*k-1);
	}
	trideig(n, W, x, D, E);
	double x0=(b+a)/2;
	double dx=(b-a)/2;
	double w0=b-a;
	for(int i=0; i<n; i++){
		x[i]=x0+dx*x[i];
		w[i]=w0*W[i*n]*W[i*n];
	}
	delete[] D, E, W;
}

void gaujac(int n, double *x, double *w, double a, double b){
	double *D=new double[n];
	double *E=new double[n];
	double *W=new double[n*n];
	for(int k=1; k<=n; k++){
		double c=a+b+2*k;
		D[k-1]=(b*b-a*a)/(c*(c-2)+(b*b==a*a));
		E[k-1]=2/c*sqrt(k*(k+a)*(k+b)*(k+a+b)/(c*c-1));
	}
	trideig(n, W, x, D, E);
	double w0=pow(2.0,a+b+1)*gamma(a+1)*gamma(b+1)/gamma(a+b+2);
	for(int i=0; i<n; i++){
		w[i]=w0*W[i*n]*W[i*n];
	}
	delete[] D, E, W;
}

void gaulag(int n, double *x, double *w){
	double *D=new double[n];
	double *E=new double[n];
	double *W=new double[n*n];
	for(int i=0; i<n; i++){
		D[i]=2*i+1;
		E[i]=i+1;
	}
	trideig(n, W, x, D, E);
	for(int i=0; i<n; i++){
		w[i]=W[i*n]*W[i*n];
	}
	delete[] D, E, W;
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
	double dx=M_SQRT2*sigma;
	double w0=sqrt(2*M_PI);
	for(int i=0; i<n; i++){
		x[i]=mu+dx*x[i];
		w[i]=w0*W[i*n]*W[i*n];
	}
	delete[] D, E, W;
}


#endif /* INTEGRATION_H_ */
