/*
 * linalg.h
 *
 *  Created on: Aug 2, 2016
 *      Author: pbrubeck
 */

#ifndef LINALG_H_
#define LINALG_H_


#include "cublas_v2.h"
#include "cusolverDn.h"
#include "mkl.h"

void trideig(int n, double *Z, double *W, double *D, double *E){
	char jobvz='V', range='A';
	int info=1, il=1, iu=n, m=n, ldz=n, lwork=20*n, liwork=10*n;
	double vl=0.0, vu=1.0, abstol=0.0;

	int *ISUPPZ=new int[2*m];
	int *IWORK=new int[liwork];
	double *WORK=new double[lwork];

	dstevr(&jobvz,&range,&n,D,E,&vl,&vu,&il,&iu,&abstol,&m,W,Z,&ldz,ISUPPZ,WORK,&lwork,IWORK,&liwork,&info);
	if(info>0){
		fprintf(stderr, "trideig failed to compute eigenvalues.\n");
	}
}

void eig(int n, double *VR, int ldvr, double *WR, double *WI, double *A, int lda){
	char jobvl='N', jobvr='V';
	int info=1, ldvl=0;
	double *VL=new double[n*ldvl];

	int lwork=8*n;
	double *WORK=new double[lwork];
	dgeev(&jobvl, &jobvr, &n, A, &lda, WR, WI, VL, &ldvl, VR, &ldvr, WORK, &lwork, &info);
	if(info>0){
		fprintf(stderr, "eig failed to compute eigenvalues.\n");
	}
}

void schur(int n, double *A, int lda, double *VS, int ldvs){
	char jobvs='V', sort='N';
	int info=1, sdim=0, lwork=8*n;

	double *WR=new double[n];
	double *WI=new double[n];
	double *WORK=new double[lwork];
	int *BWORK=new int[0];

	MKL_D_SELECT_FUNCTION_2 sel=0;
	dgees(&jobvs, &sort, sel, &n, A, &lda, &sdim, WR, WI, VS, &ldvs, WORK, &lwork, BWORK, &info);

	if(info>0){
		fprintf(stderr, "schur failed to compute factorization.\n");
	}
}

void sylvester(int m, int n, double *A, int lda, double *B, int ldb, double *C, int ldc){
	double *Q1=new double[m*m];
	schur(m, A, lda, Q1, m);

	double *Q2=new double[n*n];
	schur(n, B, ldb, Q2, n);

	char trana='N', tranb='T';
	int info=1, isgn=1;

	double alpha=1, beta=0;
	dgemm("N", "N", &m, &m, &n, &alpha, Q1, &m, C, &ldc, &beta, C, &ldc);
	dgemm("N", "T", &m, &n, &n, &alpha, C, &ldc, Q2, &n, &beta, C, &ldc);
	dtrsyl(&trana, &tranb, &isgn, &m, &n, A, &lda, B, &ldb, C, &ldc, &alpha, &info);
	dgemm("T", "N", &m, &m, &n, &alpha, Q1, &m, C, &ldc, &beta, C, &ldc);
	dgemm("N", "N", &m, &n, &n, &alpha, C, &ldc, Q2, &n, &beta, C, &ldc);

	if(info>0){
		fprintf(stderr, "sylvester failed to compute solution.\n");
	}
}


#endif /* LINALG_H_ */
