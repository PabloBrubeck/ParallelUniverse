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
void addIncrements(unsigned int* const d_in, unsigned int* const d_sum, const size_t length, const int grid){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	if(gid<length){
		d_in[gid]+=d_sum[blockIdx.x*grid/gridDim.x];
	}
}
__global__
void exclusiveSum(unsigned int *d_out, unsigned int *d_in, unsigned int *d_sum, int n){  
	extern __shared__ unsigned int temp[];
	int tid=threadIdx.x;
	int gid=blockDim.x*blockIdx.x+tid;
	temp[2*tid]=(2*gid<n)? d_in[2*gid]: 0u;  
	temp[2*tid+1]=(2*gid+1<n)? d_in[2*gid+1]: 0u;
	unsigned int offset=1u;
	unsigned int p=2u*blockDim.x;
	//downsweep
	for(unsigned d=p>>1; d>0; d>>=1){
		__syncthreads();
		if(tid<d){
			int ai=offset*(2*tid+1)-1;
			int bi=offset*(2*tid+2)-1;
			temp[bi]+=temp[ai];
		}
		offset<<=1;
	}
	//clear the last element
	if(tid==0){
		d_sum[blockIdx.x]=temp[p-1];
		temp[p-1]=0; 
	} 
	//upsweep
	for(unsigned d=1; d<p; d<<=1){
		offset>>=1;
		__syncthreads();
		if(tid<d){
			int ai=offset*(2*tid+1)-1;  
			int bi=offset*(2*tid+2)-1;
			unsigned int t=temp[ai];  
			temp[ai]=temp[bi];  
			temp[bi]+=t;
		}
	}
	__syncthreads();
	//write results to device memory 
	if(2*gid<n){
		d_out[2*gid]=temp[2*tid];
	}
	if(2*gid+1<n){
		d_out[2*gid+1]=temp[2*tid+1];
	}
}
void exclusiveScan(unsigned int* const d_in, const size_t length){
	unsigned int *d_sum;
	int n=(length+1)/2;
	int block=min(MAXTHREADS, nextPowerOf2(n));
    int grid=(n+block-1)/block;
	checkCudaErrors(cudaMalloc((void**)&d_sum, grid*sizeof(unsigned int)));
	exclusiveSum<<<grid, block, 2*block*sizeof(unsigned int)>>>(d_in, d_in, d_sum, length);
	if(grid>1){
		exclusiveScan(d_sum, grid);
		int b=min(MAXTHREADS, nextPowerOf2(length));
		int g=(length+b-1)/b;
		addIncrements<<<g, b>>>(d_in, d_sum, length, grid);
	}
	cudaFree(d_sum);
}

__global__
void getPredicate(unsigned int *d_in, unsigned int *d_scatter, size_t length, int radix, int shift){
    int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<length){ 
		int digit=(d_in[gid]>>shift)&(radix-1);
		d_scatter[digit*length+gid]=1u;
    }
}
__global__
void radixScatter(unsigned int *d_in, unsigned int *d_out, unsigned int *d_scatter, size_t length, int radix, int shift){
	int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<length){
		int digit=(d_in[gid]>>shift)&(radix-1);
		int pos=d_scatter[digit*length+gid];
		d_out[pos]=d_in[gid];
    }
}
void radixSort(unsigned int *d_in, size_t length, int radix){
	unsigned int *d_out, *d_scatter;
	checkCudaErrors(cudaMalloc((void**)&d_out, length*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&d_scatter, radix*length*sizeof(unsigned int)));
    int block=min(MAXTHREADS, nextPowerOf2(length));
    int grid=(length+block-1)/block;
	int jump=(int)log2(radix);
    for(int i=0; i<8*sizeof(unsigned int); i+=jump){
		checkCudaErrors(cudaMemset(d_scatter, 0u, radix*length*sizeof(unsigned int)));
		getPredicate<<<grid, block>>>(d_in, d_scatter, length, radix, i);
		exclusiveScan(d_scatter, radix*length);
		radixScatter<<<grid, block>>>(d_in, d_out, d_scatter, length, radix, i);
		checkCudaErrors(cudaMemcpy(d_in, d_out, length*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    }
	cudaFree(d_out);
	cudaFree(d_scatter);
}

int main(){
	//Input parameters
	int n=1024*1024;
	int radix=2;

	//Generate a random device array of size n
	unsigned int *d_input;
	checkCudaErrors(cudaMalloc((void**)&d_input, n*sizeof(unsigned int)));
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandGenerate(generator, d_input, n);

	//Set timer
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