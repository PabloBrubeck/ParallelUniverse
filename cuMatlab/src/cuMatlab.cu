#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "cuMatlab.h"
#include "examples.h"
#include "animation.h"

using namespace std;

void imageExample(const int w, const int h){
	uchar4 *d_rgba;
	cudaMalloc((void**)&d_rgba, w*h*sizeof(uchar4));

	auto dcolor = [] __device__ (double x, double y){
		cuDoubleComplex z=make_cuDoubleComplex(x, y);
		cuDoubleComplex w=asin(z);
		return hsv2rgb(angle(w)/(2*pi),1,1);
	};

	double L=3*pi;
	double xmin=-L, xmax=L, ymin=-h*L/w, ymax=h*L/w;
	cudaMap(dcolor, w, h, d_rgba, w, xmin, xmax, ymax, ymin);

	string path="/home/pbrubeck/ParallelUniverse/cuMatlab/data/DomainColor.png";
	imwrite(w,h,d_rgba,path);
	cudaFree(d_rgba);
}


__global__ void mandelbrotf(int m, int n, uchar4* d_rgba, float x1, float x2, float y1, float y2){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<m && j<n){
		float x=x1+i*(x2-x1)/(m-1);
		float y=y1+j*(y2-y1)/(n-1);
		cuComplex z=make_cuComplex(x,y);
		cuComplex w=make_cuComplex(x,y);
		int k=0;
		while(k<64 && w.x*w.x+w.y*w.y<4){
			w=cuCfmaf(w,w,z);
			k++;
		};
		float h=(k%65)/65.f;
		d_rgba[j*m+i]=jet(h);
	}
}

void fractal(int w, int h, uchar4* d_rgba, double x1, double x2, double y1, double y2){
	mandelbrotf<<<grid(w,h), MAXTHREADS>>>(w, h, d_rgba, x1, x2, y2, y1);
}


int main(int argc, char **argv){
	//animation(argc, argv, 1024, 1024, fractal);
	//auto f=[] __device__ (double x){return sinpi(x);};
	//poisson(f, -1, 1, 32);

	waveExample(512);

	printf("Program terminated.\n");
	return 0;
}
