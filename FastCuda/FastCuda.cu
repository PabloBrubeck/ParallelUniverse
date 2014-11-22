
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>
#include <ctime>
#include <curand.h>

#define MAXTHREADS 512u
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace std;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void printArray(float* arr, int n){
	for(int i=0; i<n; i++){
		printf("%.4f ", arr[i]);
	}
	printf("\n");
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
void printFromDevice(float* d_array, int length){
    float *h_temp=new float[length];
    cudaMemcpy(h_temp, d_array, length*sizeof(float), cudaMemcpyDeviceToHost); 
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
void reduceSum(unsigned int* d_in, unsigned int* d_out, const size_t elements)
{   
    int tid=threadIdx.x;
    int gid=blockIdx.x*blockDim.x+tid;
    extern __shared__ unsigned int shared[];
	shared[tid]= gid<elements? d_in[gid]: 0;
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid<s && gid<elements){
            shared[tid]=shared[tid]+shared[tid+s];
        }
        __syncthreads();
    }
    if(tid==0){
        d_out[blockIdx.x]=shared[0];
    }
}
void getSum(unsigned int *d_in, unsigned int *d_out, int n, cudaStream_t stream){
	int grid, block=MAXTHREADS;
	do{
		grid=(n+block-1)/block;
		if(grid==1){
			block=nextPowerOf2(n);
		}
		reduceSum<<<grid, block, block*sizeof(unsigned int), stream>>>(d_in, d_in, n);
		n=grid;
	}while(grid>1);
	cudaMemcpy(d_out, d_in, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
}
void partialSums(unsigned int *d_in, unsigned int *d_out, int length, int range){
	int n=length/range;
	for(int i=0; i<n; i++){
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		getSum(&(d_in[i*range]), &(d_out[i]), range, stream);
	}
	cudaDeviceSynchronize(); 
	checkCudaErrors(cudaGetLastError());
}

__global__
void histogram(unsigned int* d_in, unsigned int* d_hist, int n){
	int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<n){
		int bin=(d_in[gid]/10)%100;
		atomicAdd(&(d_hist[bin]), 1);
	}
}

int main(){
	//Input parameters
	int n=1024*1024, numBins=100;
	
	//Generate a random host array of size n and copy it to device
	unsigned int *d_in, *d_hist, *d_out;
	checkCudaErrors(cudaMalloc((void**)&d_in, n*sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_hist, numBins*sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_hist, 0, numBins*sizeof(unsigned int)));
	curandGenerator_t gen;
	curandCreateGenerator(&gen , CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(gen , 1234ULL);
	curandGenerate(gen, d_in, n);
	curandDestroyGenerator(gen);
	
	//Compute the fast histogram
	int block=min(MAXTHREADS, nextPowerOf2(n));
	int grid=(n+block-1)/block;
	histogram<<<grid, block>>>(d_in, d_hist, n);

	checkCudaErrors(cudaMalloc((void**)&d_out, 10*sizeof(unsigned int)));
	partialSums(d_hist, d_out, numBins, 10);
	printFromDevice(d_out, 10);

	cudaFree(d_in);
	cudaFree(d_hist);
	cudaFree(d_out);
    return 0;
}