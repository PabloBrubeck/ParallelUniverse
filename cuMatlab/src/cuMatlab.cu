#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "cuMatlab.h"
#include "examples.h"
#include "vertex.h"
#include "geometry.h"

using namespace std;


__global__ void mandelbrotd(int m, int n, uchar4* rgba, double x1, double x2, double y1, double y2){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<m && j<n){
		double x=x1+i*(x2-x1)/(m-1);
		double y=y1+j*(y2-y1)/(n-1);
		cuDoubleComplex z=make_cuDoubleComplex(x,y);
		cuDoubleComplex w=make_cuDoubleComplex(x,y);
		int k=0;
		while(k<64 && w.x*w.x+w.y*w.y<4){
			w=cuCfma(w,w,z);
			k++;
		};
		float t=k/64.f;
		rgba[j*m+i]=cold(t);
	}
}

void fractal(int w, int h, uchar4* d_rgba, double x1, double x2, double y1, double y2){
	static const dim3 block(32, 16);
	static const dim3 grid(ceil(w,block.x), ceil(h,block.y));
	mandelbrotd<<<grid, block>>>(w, h, d_rgba, x1, x2, y2, y1);
	cudaThreadSynchronize();
	checkCudaErrors(cudaGetLastError());
}

__global__ void complexPlotd(int m, int n, uchar4* rgba, float x1, float x2, float y1, float y2){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<m && j<n){
		double x=x1+i*(x2-x1)/(m-1);
		double y=y1+j*(y2-y1)/(n-1);
		cuDoubleComplex z=make_cuDoubleComplex(x,y);
		cuDoubleComplex w=sin(z);
		float h=angle(w)/(2*pi);
		float v=abs(w); v=0.3+0.7*(v-floor(v));
		rgba[j*m+i]=hsv2rgb(h, 1, v);
	}
}

void complexPlot(int w, int h, uchar4* rgba, double x1, double x2, double y1, double y2){
	complexPlotd<<<grid(w,h), MAXTHREADS>>>(w, h, rgba, x1, x2, y2, y1);
	cudaThreadSynchronize();
	checkCudaErrors(cudaGetLastError());
}

__global__ void computeGamma(int m, int n, uchar4* rgba, int N, double *logt, double *w, float x1, float x2, float y1, float y2){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<m && j<n){
		double x=x1+i*(x2-x1)/(m-1);
		double y=y1+j*(y2-y1)/(n-1);
		cuDoubleComplex g=make_cuDoubleComplex(0,0);
		cuDoubleComplex zm=make_cuDoubleComplex(x-1,y);
		for(int k=0; k<N; k++){
			g=g+w[k]*exp(logt[k]*zm);
		}
		float h=angle(g)/(2*pi);
		float v=abs(g); v=0.5f+0.5f*(v-floor(v));
		rgba[j*m+i]=hsv2rgb(h, 1, v);
	}
}

void complexGamma(int m, int n, uchar4* rgba, double x1, double x2, double y1, double y2){
	static int N=32;
	static double *t, *w;
	if(!t){
		cudaMallocManaged((void**)&t, N*sizeof(double));
		cudaMallocManaged((void**)&w, N*sizeof(double));
		cudaDeviceSynchronize();
		gaulag(N, t, w);
		cudaMap([] __device__ (double x){return log(x);}, N, t, t);
	}
	computeGamma<<<grid(m,n), MAXTHREADS>>>(m,n,rgba,N,t,w,x1,x2,y2,y1);
	cudaDeviceSynchronize();
}

void makeSpheres(dim3 mesh, float4* vertex, float4* norm, uchar4* color, uint4* index){
	static float4 *pos=NULL;
	static dim3 grid1=grid(mesh.z);
	static dim3 grid3, block3;
	if(pos==NULL){
		gridblock(grid3, block3, mesh);
		cudaMalloc((void**)&pos, mesh.z*sizeof(float4));
		circle<<< grid1, MAXTHREADS>>>(mesh.z, pos, 1.f);
		spheres<<<grid3, block3>>>(mesh, pos, vertex, norm, 0.1f);
		indexS2<<<grid3, block3>>>(mesh, index);
		colorSurf<<<grid3, block3>>>(mesh, color);
	}
}



int main(int argc, char **argv){
	dim3 mesh(1<<8, 1<<8, 1<<5);
	vertex(argc, argv, mesh, makeSpheres);
	//waveExample(1024);
	//auto f=[] __device__ (double x){return sinpi(x);};
	//poisson(f, -1, 1, 32);

	printf("Program terminated.\n");
	return 0;
}
