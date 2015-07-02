#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuColor.h"
#include <helper_cuda.h>
#include <cuComplex.h>

#define MAXTHREADS 512
#define COLORDEPTH 64

float2 axes=make_float2(4.0f, 4.0f);
float2 origin=make_float2(-2.0f, -2.0f);

__device__ int mandelbrot(float x, float y){
	cuComplex c=make_cuComplex(x, y);
	cuComplex z=make_cuComplex(x, y);
	float z2=0.f;
	int k=0;
	while(z2<4.f && k<COLORDEPTH){
		z=cuCaddf(cuCmulf(z, z), c);
		z2=z.x*z.x+z.y*z.y;
		k++;
	}
	return k-1;
}

__global__ void kernel(uchar4* d_pixel, uchar4 *d_cmap, float2 origin, float2 axes, uint2 image, float time){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<image.x && j<image.y){
		int gid=(image.y-j-1)*image.x+i;
		float x=fma((float)i/image.x, axes.x, origin.x);
		float y=fma((float)j/image.y, axes.y, origin.y);
		int k=mandelbrot(x, y);
		d_pixel[gid]=d_cmap[k%COLORDEPTH];
	}
}

inline int ceil(int num, int den){
	return (num+den-1)/den;
}

uchar4 *d_cmap;
void init_kernel(uint2 image){
	checkCudaErrors(cudaMalloc((void**)&d_cmap, COLORDEPTH*sizeof(uchar4)));
	jet<<<1, COLORDEPTH>>>(d_cmap, COLORDEPTH);
}

void launch_kernel(uchar4* d_pixel, uint2 image, float time){
	static const dim3 block(MAXTHREADS);
	static const dim3 grid(ceil(image.x, block.x), ceil(image.y, block.y));

	kernel<<<grid, block>>>(d_pixel, d_cmap, origin, axes, image, time);
	cudaThreadSynchronize();
	checkCudaErrors(cudaGetLastError());
}
