#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void disp(float* A, int m, int n){
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			printf("%.4f\t", A[i*m+j]);
		}
		printf("\n");
	}
}


int factorial(int n){
	int fact=1;
	for(int i=1; i<=n; i++){
		fact*=i;
	}
	return fact;
}

inline __device__ __host__
float horner(float* p, int n, float x){
	float y=0.f;
	for(int i=n; i>=0; i--){
		y=y*x+p[i];
	}
	return y;
}

void derive(float* d, float* p, int n, int k=1){
	for(int i=0; i<=n-k; i++){
		int c=1;
		for(int j=i+1; j<=i+k; j++){
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
void legendreA(float* Pml, int m, int l){
	int n=l+1;
	float* P=new float[n*n];
	legendre(P, n);
	derive(Pml, P+n*l, n, m);
}