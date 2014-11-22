#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>
#include <curand.h>

#define MAXTHREADS 512u
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace std;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void printArray(unsigned int* arr, int n){
	for(int i=0; i<n; i++){
		printf("%u ", arr[i]);
	}
	printf("\n");
}
void printFromDevice(unsigned int* d_array, int length){
    unsigned int *h_temp=new unsigned int[length];
    cudaMemcpy(h_temp, d_array, length*sizeof(unsigned int), cudaMemcpyDeviceToHost); 
    printArray(h_temp, length);
    delete[] h_temp;
}

unsigned int nextPowerOf2(unsigned int n){
  unsigned k=0;
  if(n&&!(n&(n-1))){
	  return n;
  }
  while(n!=0){
    n>>=1;
    k++;
  }
  return 1<<k;
}

__global__
void getIncrements(unsigned int* const d_in, unsigned int* const d_sum, const int grid, const int block){
	int bid=blockIdx.x*blockDim.x+threadIdx.x;
	int gid=bid*block-1;
	if(bid<grid){
		d_sum[bid]=gid<0? 0: d_in[gid];
	}
}
__global__
void addIncrements(unsigned int* const d_in, unsigned int* const d_sum, const size_t length){
	int bid=blockIdx.x;
	int gid=bid*blockDim.x+threadIdx.x;
	if(gid<length){
		d_in[gid]+=d_sum[bid];
	}
}
__global__
void inclusiveSum(unsigned int* const d_in, const size_t length){
    int tid=threadIdx.x;
	int gid=blockIdx.x*blockDim.x+tid;
    for(unsigned int s=1; s<length; s<<=1){
		unsigned int t=d_in[gid];
		if(tid>=s && gid<length){
            t+=d_in[gid-s];
        }
		__syncthreads();
		d_in[gid]=t;
		__syncthreads();
    }
}
void inclusiveScan(unsigned int* const d_in, const size_t length){
	int block=min(MAXTHREADS, nextPowerOf2(length));
    int grid=(length+block-1)/block;
	inclusiveSum<<<grid, block>>>(d_in, length);
	if(grid>1){
		unsigned int *d_sum;
		checkCudaErrors(cudaMalloc((void**)&d_sum, grid*sizeof(unsigned int)));
		int b=min(MAXTHREADS, nextPowerOf2(grid));
		int g=(grid+b-1)/b;
		getIncrements<<<g, b>>>(d_in, d_sum, grid, block);
		inclusiveScan(d_sum, grid);
		addIncrements<<<grid, block>>>(d_in, d_sum, length);
		cudaFree(d_sum);
	}
}

__global__
void getPredicate(unsigned int* d_inputVals, unsigned int* d_scatter, size_t numElems, int radix, int shift){
    int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<numElems){ 
		int digit=(d_inputVals[gid]>>shift)&(radix-1);
		d_scatter[digit*numElems+gid]=1u;
    }
}
__global__
void radixScatter(unsigned int* d_inputVals, unsigned int* d_outputVals, unsigned int* d_scatter, size_t numElems, int radix, int shift){
	int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<numElems){
		int digit=(d_inputVals[gid]>>shift)&(radix-1);
		int i=digit*numElems+gid-1;
		int pos=i<0? 0: d_scatter[i];
		d_outputVals[pos]=d_inputVals[gid];
    }
}
void radixSort(unsigned int* d_inputVals, size_t numElems, int radix){
	unsigned int *d_outputVals, *d_scatter;
	checkCudaErrors(cudaMalloc((void**)&d_outputVals, numElems*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&d_scatter, radix*numElems*sizeof(unsigned int)));
    int block=min(MAXTHREADS, nextPowerOf2(numElems));
    int grid=(numElems+block-1)/block;
	int jump=(int)log2(radix);
    for(int i=0; i<8*sizeof(unsigned int); i+=jump){
		checkCudaErrors(cudaMemset(d_scatter, 0u, radix*numElems*sizeof(unsigned int)));
		getPredicate<<<grid, block>>>(d_inputVals, d_scatter, numElems, radix, i);
		inclusiveScan(d_scatter, radix*numElems);
		radixScatter<<<grid, block>>>(d_inputVals, d_outputVals, d_scatter, numElems, radix, i);
		checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, numElems*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    }
	cudaFree(d_outputVals);
	cudaFree(d_scatter);
}

int main(){
	//Input parameters
	int n=1024*1024;
	int radix=4;

	//Generate a random device array of size n
	unsigned int *d_input;
	checkCudaErrors(cudaMalloc((void**)&d_input, n*sizeof(unsigned int)));
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandGenerate(generator, d_input, n);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//Perfom the radix sort
	radixSort(d_input, n, radix);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the sort: %f ms\n", time);

	//printFromDevice(d_input, n);
	cudaFree(d_input);
	return 0;
}
