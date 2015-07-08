#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ template <typename T>
void eulerStep(T* d_sk, T* d_y, T* d_k, float h, float w, auto f, int n){
	int gid=blockIdx.x*blockDim.x+threadTdx.x;
	if(gid<n){
		d_temp[gid]=d_y[gid]+w*d_k[gid];
		d_k[gid]=h*f(t+w*h, *d_temp, gid);
	}
}

void rungeKutta(){
	int m=1000;
	float a=0, b=10;
	float h=(b-a)/m;
	float t=a;

	lambda f;

	int n=1<<10;
	float3 *d_y, *d_k, *d_sk;

	while(t<b){

		eulerStep<float3>(d_sk, d_y, d_k, h, 0.0f, f, n);
		eulerStep<float3>(d_sk, d_y, d_k, h, 0.5f, f, n);
		eulerStep<float3>(d_sk, d_y, d_k, h, 0.5f, f, n);
		eulerStep<float3>(d_sk, d_y, d_k, h, 1.0f, f, n);

		t+=h;
	}
}