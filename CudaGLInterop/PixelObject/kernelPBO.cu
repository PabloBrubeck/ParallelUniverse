//kernelPBO.cu (Rob Farber)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>
#include <cuComplex.h>

#define MAXTHREADS 512

//Simple kernel writes changing colors to a uchar4 array
__global__ 
void kernel(uchar4* d_pixel, uint2 image, float time){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<image.x && j<image.y){
		int gid=j*image.x+i;

		unsigned char r = (i + (int)time) & 0xff;
		unsigned char g = (j + (int)time) & 0xff;
		unsigned char b = ((i + j) + (int)time) & 0xff;

		// Each thread writes one pixel location in the texture (textel)
		d_pixel[gid] = make_uchar4(r, g, b, 255u);
	}
}

inline uint ceil(uint num, uint den){
	return (num+den-1u)/den;
}


void launch_kernel(uchar4* d_pixel, uint2 image, float time){
	static const dim3 block(MAXTHREADS, 1);
	static const dim3 grid(ceil(image.x, block.x), ceil(image.y, block.y));

	kernel<<<grid, block>>>(d_pixel, image, time);

	cudaThreadSynchronize();
	checkCudaErrors(cudaGetLastError());
}