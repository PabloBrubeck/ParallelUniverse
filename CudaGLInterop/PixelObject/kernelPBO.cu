// Adapted form Rob Farber's code on drdobbs.com

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cuComplex.h>

#define MAXTHREADS 512

__device__
void mandelbrot(uchar4 &pixel, uint2 image, int i, int j){
	float x=2.f*(float(2*i)/image.x-1.f);
	float y=2.f*(float(2*j)/image.y-1.f);
	cuComplex c=make_cuComplex(x, y);
	cuComplex z=make_cuComplex(x, y);
	float z2=0.f;
	int k=0;
	do{
		z=cuCaddf(cuCmulf(z,z),c);
		z2=z.x*z.x+z.y*z.y;
		k++;
	}while(z2<4.f && k<256);
	pixel.x=(k)&0xff;
	pixel.y=(k)&0xff;
	pixel.z=(k)&0xff;
	pixel.w=0xff;
}

//Simple kernel writes changing colors to a uchar4 array
__global__ 
void kernel(uchar4* d_pixel, uint2 image, float time){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<image.x && j<image.y){
		int gid=j*image.x+i;
		mandelbrot(d_pixel[gid], image, i, j);
	}
}

inline int ceil(int num, int den){
	return (num+den-1)/den;
}


void launch_kernel(uchar4* d_pixel, uint2 image, float time){
	static const dim3 block(MAXTHREADS);
	static const dim3 grid(ceil(image.x, block.x), ceil(image.y, block.y));

	kernel<<<grid, block>>>(d_pixel, image, time);

	cudaThreadSynchronize();
	checkCudaErrors(cudaGetLastError());
}