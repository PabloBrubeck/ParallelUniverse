//kernelPBO.cu (Rob Farber)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <helper_cuda.h>

//Simple kernel writes changing colors to a uchar4 array
__global__ 
void kernel(uchar4* pos, unsigned int width, unsigned int height,
	float time)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int x = index%width;
	unsigned int y = index/width;

	if (index < width*height) {
		unsigned char r = (x + (int)time) & 0xff;
		unsigned char g = (y + (int)time) & 0xff;
		unsigned char b = ((x + y) + (int)time) & 0xff;

		// Each thread writes one pixel location in the texture (textel)
		pos[index] = make_uchar4(r,g,b,0);
	}
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(uchar4* pos, unsigned int image_width,
	unsigned int image_height, float time)
{
	// execute the kernel
	int nThreads = 256;
	int totalThreads = image_height * image_width;
	int nBlocks = totalThreads / nThreads;
	nBlocks += ((totalThreads%nThreads)>0) ? 1 : 0;

	kernel <<<nBlocks, nThreads>>>(pos, image_width, image_height, time);

	// make certain the kernel has completed 
	cudaThreadSynchronize();
	checkCudaErrors(cudaGetLastError());
}