/*
 * SpectralMethods.h
 *
 *  Created on: May 25, 2016
 *      Author: pbrubeck
 */

#ifndef SPECTRALMETHODS_H_
#define SPECTRALMETHODS_H_

#include "kernel.h"

__global__
void chebNodes(double *d_x, int n){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	if(gid<n){
		d_x[gid]=cospi(gid/(n-1.0));
	}
}

__global__
void chebDelem(double *d_D, double *d_x, int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<n && j<n){
		int gid=j*n+i;
		if(i!=j){
			int ci=1+(i==0 || i==n-1);
			int cj=1+(j==0 || j==n-1);
			d_D[gid]=(ci*((i+j)&1?-1:1))/(cj*(d_x[i]-d_x[j]));
		}else if(j>0 && j<n-1){
			d_D[gid]=-d_x[j]/(2*(1-d_x[j]*d_x[j]));
		}else{
			d_D[gid]=(j==0?1:-1)*(2*(n-1)*(n-1)+1.0)/6;
		}
	}
}

void chebD(double *d_D, double *d_x, int n){
	chebNodes<<<grid(n), MAXTHREADS>>>(d_x, n);
	chebDelem<<<grid(n,n), MAXTHREADS>>>(d_D, d_x, n);
}

void chebfftD(double *u, int length){

}

void chebfttD2(double *u, int length){

}

void fftD(double *u, int length, int order){

}

#endif /* SPECTRALMETHODS_H_ */
