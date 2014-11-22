#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>
#include <ctime>

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
void displace(unsigned int* const d_in, const size_t lenght)
{
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int temp=gid>0? d_in[gid-1]:0;
	__syncthreads();
	if(gid<lenght){
		d_in[gid]=temp;
	}
}
__global__
void getIncrements(unsigned int* const d_in, unsigned int* const d_sum, const int grid, const int block)
{
	int bid=blockIdx.x*blockDim.x+threadIdx.x;
	int gid=bid*block-1;
	if(bid<grid){
		d_sum[bid]=gid<0? 0: d_in[gid];
	}
}
__global__
void addIncrements(unsigned int* const d_in, unsigned int* const d_sum, const size_t lenght)
{
	int bid=blockIdx.x;
	int gid=bid*blockDim.x+threadIdx.x;
	if(gid<lenght){
		d_in[gid]+=d_sum[bid];
	}
}

__global__
void inclusiveSum(unsigned int* const d_in, const size_t lenght)
{
    int tid=threadIdx.x;
	int gid=blockIdx.x*blockDim.x+tid;
    for(unsigned int s=1; s<lenght; s<<=1){
		unsigned int t=d_in[gid];
		if(tid>=s && gid<lenght){
            t+=d_in[gid-s];
        }
		__syncthreads();
		d_in[gid]=t;
		__syncthreads();
    }
}
void inclusiveScan(unsigned int* const d_in, const size_t lenght){
	int block=min(MAXTHREADS, nextPowerOf2(lenght));
    int grid=(lenght+block-1)/block;
	inclusiveSum<<<grid, block>>>(d_in, lenght);
	if(grid>1){
		unsigned int *d_sum;
		checkCudaErrors(cudaMalloc((void**)&d_sum, grid*sizeof(unsigned int)));
		int b=min(MAXTHREADS, nextPowerOf2(grid));
		int g=(grid+b-1)/b;
		getIncrements<<<g, b>>>(d_in, d_sum, grid, block);
		inclusiveScan(d_sum, grid);
		addIncrements<<<grid, block>>>(d_in, d_sum, lenght);
		cudaFree(d_sum);
	}
}

__global__
void exclusiveSum(unsigned int* const d_in, const size_t lenght)
{
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int s;
	for(s=1; s<lenght; s<<=1){
        if((gid+1)%(2*s)==0 && gid<lenght){
            d_in[gid]+=d_in[gid-s];
        }
        __syncthreads();
    }
	if(gid==lenght-1){
		d_in[gid]=0;
	}
	__syncthreads();
	for(; s>0; s>>=1){
        if((gid+1)%(2*s)==0 && gid<lenght){
			unsigned int right=d_in[gid];
            d_in[gid]+=d_in[gid-s];
			d_in[gid-s]=right;
        }
        __syncthreads();
    }
}
void exclusiveScan(unsigned int* const d_in, const size_t lenght){
	int block=min(MAXTHREADS, nextPowerOf2(lenght));
    int grid=(lenght+block-1)/block;
	inclusiveScan(d_in, lenght);
	displace<<<grid, block>>>(d_in, lenght);
}

int main(){
	//Input parameters
	int n=1024*1024, numBins=1024;
	
	//Generate a random host array of size n and copy it to device
	float *d_in, *h_in=new float[n];
	srand((unsigned long)time(NULL));
	for(int i=0; i<n; i++){
		h_in[i]=1-(float)(rand()%10001)/5000;
	}
	cudaMalloc((void**)&d_in, n*sizeof(float));
	cudaMemcpy(d_in, h_in, n*sizeof(float), cudaMemcpyHostToDevice);

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
	delete[] h_in;
	return 0;
}