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

__host__ __device__ float LegendreP(float* a, int n, int m, float x){
	float temp, yy=0;
	float y=(n>m+1)?a[n-1]:0;
	for(int k=n-2; k>m; k--){
		temp=y;
		y=a[k]+(2*k+1)*x*y/(k-m+1)-(k+m+1)*yy/(k-m+2);
		yy=temp;
	}
	int prod=1;
	for(int i=1; i<2*m; i+=2){
		prod*=i;
	}
	float p0=(m%2==0?1:-1)*prod*powf(1-x*x,m/2.0);
	return p0*(a[m]+(2*m+1)*(x*y-yy/2));
}
