/*
 * LinearAlgebra.h
 *
 *  Created on: May 25, 2016
 *      Author: pbrubeck
 */

#ifndef LINEARALGEBRA_H_
#define LINEARALGEBRA_H_

#include "cublas_v2.h"
#include "cusolverDn.h"

void trideigs(double *Z, double *W, double *D, double *E, int n){

}

void sylvester(double *X, double *A, double *B, double *C, int m, int n){
	double *a=new double[m];
	double *S=new double[m*m];
	//eigs(a,S,A,m);

	double *b=new double[n];
	double *P=new double[n*n];
	//eigs(b,P,B,n);

	//dtrsm('L',C,S,m,n);
	//dtrmm('R',C,P,m,n);

	// some magic occurs X

	//dtrmm('L',X,S,m,n);
	//dtrsm('R',X,P,m,n);

}


#endif /* LINEARALGEBRA_H_ */
