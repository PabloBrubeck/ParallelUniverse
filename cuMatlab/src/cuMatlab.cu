#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "cuMatlab.h"
#include "examples.h"

using namespace std;


void imageExample(const int w, const int h){
	uchar4 *d_out;
	cudaMalloc((void**)&d_out, w*h*sizeof(uchar4));

	auto lambda = [] __device__ (float x, float y){
		cuComplex w=make_cuComplex(sin(x)*cosh(y), cos(x)*sinh(y));
		float h=atan2f(w.y, w.x)/(2*pi);
		return hsv2rgb(h,1,1);
	};

	float xmin=-3*pi, xmax=3*pi, ymin=-4, ymax=4;
	cudaMap(lambda, w, h, d_out, w, xmin, xmax, ymin, ymax);

	string path="/home/pbrubeck/ParallelUniverse/cuMatlab/doc/DomainColor.png";
	imwrite(w,h,d_out,path);

	printf("Image saved!\n");
	cudaFree(d_out);
}

void fractalExample(const int w, const int h){
	uchar4 *d_out;
	cudaMalloc((void**)&d_out, w*h*sizeof(uchar4));

	auto lambda = [] __device__ (float x, float y){
		cuComplex z=make_cuComplex(x,y);
		cuComplex w=make_cuComplex(x,y);
		float w2=0;
		int i=0;
		do{
			w=cuCfmaf(w,w,z);
			w2=w.x*w.x+w.y*w.y;
			i++;
		}while(i<64 && w2<4);
		float h=i/63.0;
		return jet(h);
	};

	float xmin=-(2.f*w)/h, xmax=(2.f*w)/h, ymin=-2.f, ymax=2.f;
	cudaMap(lambda, w, h, d_out, w, xmin, xmax, ymin, ymax);

	string path="/home/pbrubeck/ParallelUniverse/cuMatlab/doc/Fractal.png";
	imwrite(w,h,d_out,path);

	printf("Fractal completed!\n");
	cudaFree(d_out);
}

int main(int argc, char **argv){
	fractalExample(4*1920, 4*1080);
	return 0;
}
