#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>
#include <ctime>

#define MAXTHREADS 512

using namespace std;

void printArray(float* arr, int n){
	for(int i=0; i<n; i++){
		printf("%.4f ", arr[i]);
	}
	printf("\n");
}
void printArray(unsigned int* arr, int n){
	for(int i=0; i<n; i++){
		printf("%d ", arr[i]);
	}
	printf("\n");
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
	int dimGrid, dimBlock=MAXTHREADS;
	do{
		dimGrid=(n-1)/dimBlock+1;
		if(dimGrid==1){
			dimBlock=n;
		}
		reduceMax<<<dimGrid, dimBlock, dimBlock*sizeof(float)>>>(d_in, d_in, n);
		n=dimGrid;
	}while(dimGrid>1);
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
	int dimGrid, dimBlock=MAXTHREADS;
	do{
		dimGrid=(n-1)/dimBlock+1;
		if(dimGrid==1){
			dimBlock=n;
		}
		reduceMin<<<dimGrid, dimBlock, dimBlock*sizeof(float)>>>(d_in, d_in, n);
		n=dimGrid;
	}while(dimGrid>1);
	cudaMemcpy(h_out, d_in, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
}

__global__
void histogram(const float* const d_logLuminance, unsigned int* const d_cdf,  float h_min, float range, int n, const int numBins)
{
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n){
		int bin=(d_logLuminance[gid]-h_min)/range*numBins;
		bin=min(numBins-1, max(bin,0));
		atomicAdd(&(d_cdf[bin]), 1);
	}
}

__global__
void scan(unsigned int* const d_cdf, const size_t numBins)
{
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    for(unsigned int s=1; s<numBins; s*=2){
        if(gid>=s && gid<numBins){
            d_cdf[gid]+=d_cdf[gid-s];
        }
        __syncthreads();
    }
}

int main(){
	//Input parameters
	int n=98304, numBins=1024;

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
	int dimBlock=MAXTHREADS;
	int dimGrid=(n-1)/dimBlock+1;
	histogram<<<dimGrid, dimBlock>>>(d_in, d_hist, *h_min, range, n, numBins);

	//Calculate the cumulative distribution function
	scan<<<1, numBins>>>(d_hist, numBins);

	//Copy the CDF array back to host for display
	unsigned int *h_hist=new unsigned int[numBins];
	cudaMemcpy(h_hist, d_hist, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	printArray(h_hist, numBins);
	
	//Free memory
	cudaFree(d_in);
	cudaFree(d_hist);
	delete h_min;
	delete h_max;
	delete[] h_in;
	delete[] h_hist;
	return 0;
}