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
//#include "mkl.h"

void trideigs(double *Z, double *W, double *D, double *E, int N){
	/*double VL=0.0, VU=1.0, ABSTOL=0.0;
	int IL=1, IU=N, M=N, LDZ=N, LWORK=20*N, LIWORK=10*N, INFO;

	int *ISUPPZ=(int*)malloc(2*M*sizeof(int));
	int *IWORK=(int*)malloc(LIWORK*sizeof(int));
	W=(double*)malloc(N*sizeof(double));
	Z=(double*)malloc(LDZ*M*sizeof(double));
	double *WORK=(double*)malloc(LWORK*sizeof(double));

	//dstevr("V","A",&N,D,E,&VL,&VU,&IL,&IU,&ABSTOL,&M,W,Z,&LDZ,ISUPPZ,WORK,&LWORK,IWORK,&LIWORK,&INFO);
	if(INFO>0){
		fprintf(stderr, "The algorithm failed to compute eigenvalues.\n");
	}
	free((void*)WORK);
	free((void*)IWORK);*/
}

void sylvester(double *A, double *B, double *C, int m, int n, int ldc){
	double *a=new double[m];
	double *S=new double[m*m];
	//eigs(a,S,A,m);

	double *b=new double[n];
	double *P=new double[n*n];
	//eigs(b,P,B,n);

	//dtrsm('L',C,S,m,n);
	//dtrmm('R',C,P,m,n);

	for(int i=0; i<m; i++){
		for(int j=0; i<n; j++){
			C[ldc*j+i]/=a[i]+b[j];
		}
	}

	//dtrmm('L',C,S,m,n);
	//dtrsm('R',C,P,m,n);

}


#endif /* LINEARALGEBRA_H_ */
