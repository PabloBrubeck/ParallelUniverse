#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>


void disp(float* A, int m, int n){
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			printf("%.3f\t", A[i*m+j]);
		}
		printf("\n");
	}
}


inline __device__ __host__
float horner(const float* p, float x, int n){
	float y=0.f;
	for(int i=n; i>=0; i--){
		y=y*x+p[i];
	}
	return y;
}

void derive(float* d, float* p, int n, int k=1){
	for(int i=1; i<=n-k; i++){
		int c=1;
		for(int j=1; j<=i+n-1; j++){
			c*=j;
		}
		d[i]=c*p[i+k];
	}
}

void legendre(float* P, int n){
	for(int k=0; k<=2*n; k++){
		P[k]=0.f;
	}
	P[n*0+0]=P[n*1+1]=1.f;
	for(int i=1; i<n-1; i++){
		P[n*(i+1)+0]=-i*P[n*(i-1)+0]/(i+1);
		for(int j=1; j<=n; j++){
			P[n*(i+1)+j]=((2*i+1)*P[n*(i)+(j-1)]-i*P[n*(i-1)+j])/(i+1);
		}
	}
}