#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>
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
void reduceMax(const float* const d_in, float* d_out, const size_t elements)
{   
    int tid=threadIdx.x;
    int gid=blockIdx.x*blockDim.x+tid;
    extern __shared__ float shared[];
	shared[tid]= gid<elements? d_in[gid]: -FLT_MAX;
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid<s && gid<elements){
            shared[tid]=max(shared[tid], shared[tid+s]);
        }
        __syncthreads();
    }
    if(tid==0){
        d_out[blockIdx.x]=shared[0];
    }
}
void getMax(const float* const d_data, float *h_out, int n){
	float *d_in;
	cudaMalloc((void**)&d_in, n*sizeof(float));
	cudaMemcpy(d_in, d_data, n*sizeof(float), cudaMemcpyDeviceToDevice);
	int grid, block=MAXTHREADS;
	do{
		grid=(n+block-1)/block;
		if(grid==1){
			block=nextPowerOf2(n);
		}
		reduceMax<<<grid, block, block*sizeof(float)>>>(d_in, d_in, n);
		n=grid;
	}while(grid>1);
	cudaMemcpy(h_out, d_in, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
}

__global__
void reduceMin(const float* const d_in, float* d_out, const size_t elements)
{   
    int tid=threadIdx.x;
    int gid=blockIdx.x*blockDim.x+tid;
    extern __shared__ float shared[];
	shared[tid]= gid<elements? d_in[gid]: FLT_MAX;
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid<s && gid<elements){
            shared[tid]=min(shared[tid], shared[tid+s]);
        }
        __syncthreads();
    }
    if(tid==0){
        d_out[blockIdx.x]=shared[0];
    }
}
void getMin(const float* const d_data, float *h_out, int n){
	float *d_in;
	cudaMalloc((void**)&d_in, n*sizeof(float));
	cudaMemcpy(d_in, d_data, n*sizeof(float), cudaMemcpyDeviceToDevice);
	int grid, block=MAXTHREADS;
	do{
		grid=(n+block-1)/block;
		if(grid==1){
			block=nextPowerOf2(n);
		}
		reduceMin<<<grid, block, block*sizeof(float)>>>(d_in, d_in, n);
		n=grid;
	}while(grid>1);
	cudaMemcpy(h_out, d_in, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
}

__global__
void histogram(const float* const d_in, unsigned int* const d_cdf,  float h_min, float range, int n, const int numBins)
{
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n){
		int bin=(d_in[gid]-h_min)/range*numBins;
		bin=min(numBins-1, max(bin, 0));
		atomicAdd(&(d_cdf[bin]), 1);
	}
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
void getIncrements(unsigned int* const d_in, unsigned int* const d_sum, const int grid, const int block)
{
	int bid=blockIdx.x*blockDim.x+threadIdx.x;
	int gid=(bid+1)*block-1;
	if(bid<grid){
		d_sum[bid]=d_in[gid];
	}
}
__global__
void inclusiveSum(unsigned int* const d_in, const size_t length)
{
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
		exclusiveScan(d_sum, grid);
		addIncrements<<<grid, block>>>(d_in, d_sum, length, grid);
		cudaFree(d_sum);
	}
}

int main(){
	//Input parameters
	int n=1000000, numBins=1024;
	
	//Generate a random device array of size n
	float *d_in;
	checkCudaErrors(cudaMalloc((void**)&d_in, n*sizeof(unsigned int)));
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandGenerateUniform(generator, d_in, n);

	//Get maximimum and minimum
	float *h_min=new float(), *h_max=new float();
	getMin(d_in, h_min, n);
	getMax(d_in, h_max, n);
	float range=(*h_max)-(*h_min);
	printf("min=%f max=%f range=%f\n", *h_min, *h_max, range);
	
	//Allocate a histogram array on device
	unsigned int *d_hist;	
	cudaMalloc((void**)&d_hist, numBins*sizeof(unsigned int));
	cudaMemset(d_hist, 0, numBins*sizeof(unsigned int));
	
	//Fill in the histogram
	int block=min(MAXTHREADS, nextPowerOf2(n));
    int grid=(n+block-1)/block;
	histogram<<<grid, block>>>(d_in, d_hist, *h_min, range, n, numBins);

	//Calculate and display the cumulative distribution function
	exclusiveScan(d_hist, numBins);
	printFromDevice(d_hist, numBins);
	
	//Free memory
	cudaFree(d_in);
	cudaFree(d_hist);
	delete h_min;
	delete h_max;
	return 0;
}