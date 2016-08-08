/*
 * image.h
 *
 *  Created on: Aug 1, 2016
 *      Author: pbrubeck
 */

#ifndef IMAGE_H_
#define IMAGE_H_

#include "PNG.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_math.h"

using namespace std;

__host__ __device__ static __inline__
uchar4 gray(float t){
	unsigned char v=(unsigned char)(255*clamp(t, 0.f, 1.f));
	return make_uchar4(v,v,v,255);
}

__host__ __device__ static __inline__
uchar4 jet(float t){
	unsigned char r=(unsigned char)(255*clamp(1.5f-fabsf(4*t-3), 0.f, 1.f));
	unsigned char g=(unsigned char)(255*clamp(1.5f-fabsf(4*t-2), 0.f, 1.f));
	unsigned char b=(unsigned char)(255*clamp(1.5f-fabsf(4*t-1), 0.f, 1.f));
	return make_uchar4(r,g,b,255);
}

__host__ __device__ static __inline__
uchar4 hot(float t){
	unsigned char r=(unsigned char)(255*clamp(8*t/3  , 0.f, 1.f));
	unsigned char g=(unsigned char)(255*clamp(8*t/3-1, 0.f, 1.f));
	unsigned char b=(unsigned char)(255*clamp(4*t-3  , 0.f, 1.f));
	return make_uchar4(r,g,b,255);
}

__host__ __device__ static __inline__
uchar4 cold(float t){
	unsigned char r=(unsigned char)(255*clamp(2.0f-fabsf(6*t-3), 0.f, 1.f));
	unsigned char g=(unsigned char)(255*clamp(2.0f-fabsf(6*t-2), 0.f, 1.f));
	unsigned char b=(unsigned char)(255*clamp(1.5f-fabsf(6*t-1), 0.f, 1.f));
	return make_uchar4(r,g,b,255);
}

__host__ __device__ static __inline__
uchar4 hsv2rgb(float h, float s, float v){
	h-=floorf(h);
	float c=v*s;
	unsigned char M=(unsigned char)(255*(v-c));
	unsigned char C=(unsigned char)(255*v);
	unsigned char X=(unsigned char)(255*(v-c*fabsf(fmodf(6*h,2)-1)));
	switch((int)floorf(6*h)){
		case 0 : return make_uchar4(C,X,M,255);
		case 1 : return make_uchar4(X,C,M,255);
		case 2 : return make_uchar4(M,C,X,255);
		case 3 : return make_uchar4(M,X,C,255);
		case 4 : return make_uchar4(X,M,C,255);
		case 5 : return make_uchar4(C,M,X,255);
		default: return make_uchar4(M,M,M,255);
	}
}

template<typename T>
void mat2gray(int m, int n, uchar4* rgba, T* A, int lda){
	T minval, maxval;
	minmax(&minval, &maxval, m*lda, A);
	auto lambda = [minval, maxval] __device__ (T x){
		unsigned char v=(unsigned char)(255*(x-minval)/(maxval-minval));
		return make_uchar4(v, v, v, 255);
	};
	cudaMap(lambda, m, n, A, lda, rgba, m);
}

void imwrite(int w, int h, uchar4* rgba, string path){
	unsigned char *temp=new unsigned char[4*w*h];
	cudaMemcpy(temp, rgba, w*h*sizeof(uchar4), cudaMemcpyDeviceToHost);

	PNG outPng;
	outPng.Create(w, h);
	std::copy(temp, temp+4*w*h, std::back_inserter(outPng.data));
	outPng.Save(path);
	outPng.Free();
	delete[] temp;
}

void imread(int* w, int* h, uchar4* rgba, string path){
	PNG inPng(path);
	*w=inPng.w;
	*h=inPng.h;
	int size=inPng.w*inPng.h*sizeof(uchar4);
	cudaMalloc((void**)&rgba, size);
	cudaMemcpy(rgba, &inPng.data[0], size, cudaMemcpyHostToDevice);
	inPng.Free();
}


#endif /* IMAGE_H_ */
