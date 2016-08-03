#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "cuMatlab.h"
#include "examples.h"

using namespace std;


void imageExample(const int w, const int h){
	uchar4 *d_rgba;
	cudaMalloc((void**)&d_rgba, w*h*sizeof(uchar4));

	auto lambda = [] __device__ (double x, double y){
		cuDoubleComplex z=make_cuDoubleComplex(x, y);
		cuDoubleComplex w=acos(z);
		return hsv2rgb(angle(w)/(2*pi),1,1);
	};

	double L=3*pi;
	double xmin=-L, xmax=L, ymin=-h*L/w, ymax=h*L/w;
	cudaMap(lambda, w, h, d_rgba, w, xmin, xmax, ymax, ymin);

	string path="/home/pbrubeck/ParallelUniverse/cuMatlab/data/DomainColor.png";
	imwrite(w,h,d_rgba,path);
	cudaFree(d_rgba);
}

void fractalExample(const int w, const int h){
	uchar4 *d_rgba;
	cudaMalloc((void**)&d_rgba, w*h*sizeof(uchar4));

	auto mandelbrot = [] __device__ (float x, float y){
		cuComplex z=make_cuComplex(x,y);
		cuComplex w=make_cuComplex(x,y);
		float w2=0;
		int i=0;
		do{
			w=cuCfmaf(w,w,z);
			w2=w.x*w.x+w.y*w.y;
			i++;
		}while(i<256 && w2<4);
		float h=(i%64)/64.0;
		return jet(h);
	};

	float xmin=-(2.f*w)/h, xmax=(2.f*w)/h, ymin=-2.f, ymax=2.f;
	cudaMap(mandelbrot, w, h, d_rgba, w, xmin, xmax, ymax, ymin);

	string path="/home/pbrubeck/ParallelUniverse/cuMatlab/data/Fractal.png";
	imwrite(w,h,d_rgba,path);
	cudaFree(d_rgba);
}

int main(int argc, char **argv){
	fractalExample(1<<13, 1<<13);
	printf("Program terminated.\n");
	return 0;
}
