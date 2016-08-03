/*
 * special_functions.h
 *
 *  Created on: Aug 2, 2016
 *      Author: pbrubeck
 */

#ifndef SPECIAL_FUNCTIONS_H_
#define SPECIAL_FUNCTIONS_H_

#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ __device__ static __inline__
double sinc(double x){
	return x==0?1:sin(x)/x;
}

__host__ __device__ static __inline__
double jinc(double x){
	return x==0?1:j0(x)/x;
}

/*
 * Orthogonal polynomial series using the Clenshaw algorithm
 */

__host__ __device__ static __inline__
double ChebT(int n, double* a, double x){
	double temp, yy=0;
	double y=(n>1)?a[n-1]:0;
	for(int k=n-2; k>0; k--){
		temp=y;
		y=a[k]+2*x*y-yy;
		yy=temp;
	}
	return a[0]+x*y-yy;
}

__host__ __device__ static __inline__
double LegendreP(int n, double* a, int m, double x){
	double temp, yy=0;
	double y=(n>m+1)?a[n-1]:0;
	for(int k=n-2; k>m; k--){
		temp=y;
		y=a[k]+(2*k+1)*x*y/(k-m+1)-(k+m+1)*yy/(k-m+2);
		yy=temp;
	}
	int prod=1;
	for(int i=1; i<2*m; i+=2){
		prod*=i;
	}
	double p0=(m&1?-1:1)*prod*pow(1-x*x,m/2.0);
	return p0*(a[m]+(2*m+1)*(x*y-yy/2));
}

__host__ __device__ static __inline__
double LaguerreL(int n, double* a, double alpha, double x){
	double temp, yy=0;
	double y=(n>1)?a[n-1]:0;
	for(int k=n-2; k>0; k--){
		temp=y;
		y=a[k]+(2*k+1+alpha-x)/(k+1)*y-(k+1+alpha)/(k+2)*yy;
		yy=temp;
	}
	return a[0]+(1+alpha-x)*y-(1+alpha)*yy/2;
}

__host__ __device__ static __inline__
double HermiteH(int n, double* a, double x){
	double temp, yy=0;
	double y=(n>1)?a[n-1]:0;
	for(int k=n-2; k>0; k--){
		temp=y;
		y=a[k]+2*(x*y-(k+1)*yy);
		yy=temp;
	}
	return a[0]+2*(x*y-yy);
}

__host__ __device__ static __inline__
double HermitePsi(int n, double* a, double x){
	double temp, yy=0;
	double y=(n>1)?a[n-1]:0;
	for(int k=n-2; k>0; k--){
		temp=y;
		y=a[k]+sqrt(2.0/(k+1))*x*y-sqrt((k+1.0)/(k+2))*yy;
		yy=temp;
	}
	double h0=pow(M_PI,-0.25)*exp(-x*x/2);
	return h0*(a[0]+M_SQRT2*(x*y-yy/2));
}


#endif /* SPECIAL_FUNCTIONS_H_ */
