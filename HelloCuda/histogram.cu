#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>
#include <ctime>

#define MAXTHREADS 1024u
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
void printFromDevice(unsigned int* d_array, int size){
    unsigned int *h_temp=new unsigned int[size];
    cudaMemcpy(h_temp, d_array, size*sizeof(unsigned int), cudaMemcpyDeviceToHost); 
    for(int i=0; i<size; i++){
        printf("%u ", h_temp[i]);
    }
    printf("\n");
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
	int dimGrid, dimBlock=min(MAXTHREADS, nextPowerOf2(n));;
	do{
		dimGrid=(n+dimBlock-1)/dimBlock;
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
	int dimGrid, dimBlock=min(MAXTHREADS, nextPowerOf2(n));
	do{
		dimGrid=(n+dimBlock-1)/dimBlock;
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
		bin=min(numBins-1, max(bin, 0));
		atomicAdd(&(d_cdf[bin]), 1);
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
    }
}
__global__
void getIncrements(unsigned int* const d_in, unsigned int* const d_sum, const int dimGrid, const int dimBlock)
{
	int bid=blockIdx.x*blockDim.x+threadIdx.x;
	int gid=bid*dimBlock-1;
	if(bid<dimGrid){
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

void inclusiveScan(unsigned int* const d_in, const size_t lenght){
	int dimBlock=min(MAXTHREADS, nextPowerOf2(lenght));
    int dimGrid=(lenght+dimBlock-1)/dimBlock;
	inclusiveSum<<<dimGrid, dimBlock>>>(d_in, lenght);
	if(dimGrid>1){
		unsigned int *d_sum;
		checkCudaErrors(cudaMalloc((void**)&d_sum, dimGrid*sizeof(unsigned int)));
		int b=min(MAXTHREADS, nextPowerOf2(dimGrid));
		int g=(dimGrid+b-1)/b;
		getIncrements<<<g, b>>>(d_in, d_sum, dimGrid, dimBlock);
		inclusiveScan(d_sum, dimGrid);
		addIncrements<<<dimGrid, dimBlock>>>(d_in, d_sum, lenght);
		cudaFree(d_sum);
	}
}

void exclusiveScan(unsigned int* const d_in, const size_t lenght){
	int dimBlock=min(MAXTHREADS, nextPowerOf2(lenght));
    int dimGrid=(lenght+dimBlock-1)/dimBlock;
	inclusiveScan(d_in, lenght);
	displace<<<dimGrid, dimBlock>>>(d_in, lenght);
}

__global__
void getPredicate(unsigned int* const d_inputVals, 
                  unsigned int* const d_pos0, 
				  unsigned int* const d_pos1, 
				  const size_t numElems, 
				  const int k)
{
    int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<numElems){ 
		unsigned int bit=(d_inputVals[gid]>>k)&1;
		d_pos1[gid]=bit;
		d_pos0[gid]=!bit;
    }
}

__global__
void radixScatter(unsigned int* const d_inputVals,
			   unsigned int* const d_outputVals,
			   unsigned int* const d_pos0,
			   unsigned int* const d_pos1,
			   const size_t numElems,
			   const int k)
{
	int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<numElems){
		int pos;
		unsigned int n=d_inputVals[gid];
		int bit=(n>>k)&1;
		if(bit){
			pos=gid>0? d_pos1[gid-1]: 0;
			pos+=d_pos0[numElems-1];
		}else{
			pos=gid>0? d_pos0[gid-1]: 0;
		}
		d_outputVals[pos]=d_inputVals[gid];
    }
}

void your_sort(unsigned int* const d_inputVals,
               const size_t numElems)
{
    unsigned int *d_outputVals, *d_pos0, *d_pos1;
	checkCudaErrors(cudaMalloc((void**)&d_outputVals, numElems*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&d_pos0, numElems*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&d_pos1, numElems*sizeof(unsigned int)));
    int dimBlock=min(MAXTHREADS, nextPowerOf2(numElems));
    int dimGrid=(numElems+dimBlock-1)/dimBlock;
    for(int i=0; i<8*sizeof(unsigned int); i++){
		getPredicate<<<dimGrid, dimBlock>>>(d_inputVals, d_pos0, d_pos1, numElems, i);
		inclusiveScan(d_pos0, numElems);
		inclusiveScan(d_pos1, numElems);
		radixScatter<<<dimGrid, dimBlock>>>(d_inputVals, d_outputVals, d_pos0, d_pos1, numElems, i);
		checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, numElems*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    }
	cudaFree(d_outputVals);
	cudaFree(d_pos0);
	cudaFree(d_pos1);
}


int main(){
	//Input parameters
	int n=1024*1024, numBins=2;
	
	//Generate a random host array of size n and copy it to device
	unsigned int *d_input, *h_input=new unsigned int[n];
	cudaMalloc((void**)&d_input, n*sizeof(unsigned int));
	
	srand((unsigned long)time(NULL));
	for(int i=0; i<n; i++){
		h_input[i]=(unsigned int)(rand()%1024);
	}
	cudaMemcpy(d_input, h_input, n*sizeof(unsigned int), cudaMemcpyHostToDevice);
	your_sort(d_input, n);
	printFromDevice(d_input, n);
	cudaFree(d_input);
	delete[] h_input;

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
	int dimBlock=min(MAXTHREADS, nextPowerOf2(n));
    int dimGrid=(n+dimBlock-1)/dimBlock;
	histogram<<<dimGrid, dimBlock>>>(d_in, d_hist, *h_min, range, n, numBins);

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