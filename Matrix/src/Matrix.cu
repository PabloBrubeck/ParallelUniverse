// Simple demonstration on cuBLAS usage

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cmath>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cufft.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

#define MAXTHREADS 512

int ceil(int num, int den){
	return (num+den-1)/den;
}
int nextPow2(int v){
	v--;
	v|=v>>1;
	v|=v>>2;
	v|=v>>4;
	v|=v>>8;
	v|=v>>16;
	v++;
	return v;
}

char* toString(cuDoubleComplex z){
	char* s=new char[64];
	sprintf(s, "% f%+fi", z.x, z.y);
	return s;
}
template<class T>
void printMatrix(T *A, int m, int n){
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			printf("% f\t", A[j*m+i]);
		}
		printf("\n");
	}
}
void printMatrix(cuDoubleComplex *Z, int m, int n){
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			cout<<toString(Z[j*m+i])<<"\t";
		}
		cout<<endl;
	}
}

void dftmtx(cuDoubleComplex *W, int n){
	cuDoubleComplex w=make_cuDoubleComplex(cos(2*M_PI/n), sin(2*M_PI/n));
	cuDoubleComplex wn=make_cuDoubleComplex(1.f, 0.f);
	for(int j=0; j<n; j++){
		W[j*n]=make_cuDoubleComplex(1.f, 0.f);
		for(int i=1; i<n; i++){
			W[j*n+i]=cuCmul(W[j*n+i-1], wn);
		}
		wn=cuCmul(wn, w);
	}
}
void runTest(int N){
	cublasHandle_t handle;
	cublasCreate(&handle);

	int N2=N*N;
	cuDoubleComplex *h_W=new cuDoubleComplex[N2];
	cuDoubleComplex *h_C=new cuDoubleComplex[N2];
	dftmtx(h_W, N);

	cuDoubleComplex alpha = make_cuDoubleComplex(1.f/N, 0.f);
	cuDoubleComplex beta  = make_cuDoubleComplex(0.f, 0.f);

	cuDoubleComplex *d_W, *d_C;
	checkCudaErrors(cudaMalloc((void**)&d_W, N2*sizeof(cuDoubleComplex)));
	checkCudaErrors(cudaMalloc((void**)&d_C, N2*sizeof(cuDoubleComplex)));

	cublasSetVector(N2, sizeof(cuDoubleComplex), h_W, 1, d_W, 1);
	cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, N, N, N, &alpha, d_W, N, d_W, N, &beta, d_C, N);
	cublasGetVector(N2, sizeof(cuDoubleComplex), d_C, 1, h_C, 1);

	printMatrix(h_C, N, N);
	cublasDestroy(handle);
}

__global__ void chebGrid(double *d_x, int N){
	int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<N){
		d_x[gid]=cospi(gid/(N-1.0));
	}
}
__global__ void chebDkernel(double *d_D, double *d_x, int N){
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int j=blockDim.y*blockIdx.y+threadIdx.y;
	if(i<N && j<N){
		int n=N-1;
		if(i==j){
			d_D[j*N+i]=(j%n==0)? (j==0?1:-1)*(2*n*n+1)/6.0 : d_x[j]/(2*(d_x[j]*d_x[j]-1));
		}else{
			d_D[j*N+i]=(((i%n==0)+1)*(1-2*((i+j)%2)))/(((j%n==0)+1)*(d_x[i]-d_x[j]));
		}
	}
}
__global__ void kron(double *d_A, double *d_B, double *d_C, int m, int n, int p, int q){
	int x=blockDim.x*blockIdx.x+threadIdx.x;
	int y=blockDim.y*blockIdx.y+threadIdx.y;
	if(x<m*n && y<p*q){
		int j=x/m, i=x%m;
		int l=y/p, k=y%p;
		int idx=(j*q+l)*m*p+(i*p+k);
		d_C[idx]=d_A[x]*d_B[y];
	}
}


void chebD(double *d_D, double *d_x, int N){
	dim3 block(min(nextPow2(N), MAXTHREADS));
	dim3 grid1(ceil(N, block.x));
	dim3 grid2(ceil(N, block.x), ceil(N, block.y));
	chebGrid<<<grid1, block>>>(d_x, N);
	chebDkernel<<<grid2, block>>>(d_D, d_x, N);
}

int main(int argc, char **argv){
	runTest(8);

	cublasHandle_t handle;
	cublasCreate(&handle);

	int N=8, N2=N*N;

	double *d_D, *d_D2, *d_x;
	checkCudaErrors(cudaMalloc((void**)&d_D,  N2*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_D2, N2*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_x,  N*sizeof(double)));
	chebD(d_D, d_x, N);

	double alpha=1.0;
	double beta=0.0;
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_D, N, d_D, N, &beta, d_D2, N);

	double *h_D2=new double[N2];
	cublasGetVector(N2, sizeof(double), d_D2, 1, h_D2, 1);
	printMatrix(h_D2, N, N);
}
