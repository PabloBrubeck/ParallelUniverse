#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuColor.h"
#include <helper_cuda.h>
#include <cuComplex.h>

#define MAXTHREADS 512
#define COLORDEPTH 256

uchar4 *d_cmap;
float *d_u, *d_ul, *d_lap;
double2 axes=make_double2(4.0f, 4.0f);
double2 origin=make_double2(-2.0f, -2.0f);

inline int ceil(int num, int den){
	return (num+den-1)/den;
}

__global__ void laplacian(int width, int height, float *d_lap, float *d_u){
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int j=blockDim.y*blockIdx.y+threadIdx.y;
	if(i<width && j<height){
		int ip=(i+1)%width;
		int im=(i-1+width)%width;
		int jp=(j+1)%height;
		int jm=(j-1+height)%height;
		d_lap[i*height+j]=d_u[ip*height+j]
				+d_u[im*height+j]
				+d_u[i*height+jp]
				+d_u[i*height+jm]
				-4*d_u[i*height+j];
	}
}
__global__ void heatSolve(int width, int height, uchar4 *d_pixel, uchar4 *d_cmap, float *d_u, float *d_lap, float c, double2 origin, double2 axes, float time){
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int j=blockDim.y*blockIdx.y+threadIdx.y;
	if(i<width && j<height){
		int gid=i*height+j;
		d_u[gid]+=c*d_lap[gid];
		float x=fma((float)i/width, (float)axes.x, (float)origin.x);
		float y=fma((float)j/height, (float)axes.y, (float)origin.y);
		if(x*x+y*y<1.E-2f){
			d_u[gid]=1.f;
		}
		if(i==0 || j==0 || i==width-1 || j==height-1){
			d_u[gid]=0;
		}
		int k=(int)(COLORDEPTH*d_u[gid]);
		d_pixel[gid]=d_cmap[clamp(k, 0, COLORDEPTH-1)];
	}
}
void heatPDE(int width, int height, uchar4 *d_pixel, uchar4 *d_cmap, float *d_u, float *d_lap, double2 origin, double2 axes, float time){
	static dim3 block(MAXTHREADS);
	static dim3 grid(ceil(width, block.x), ceil(height, block.y));
	laplacian<<<grid, block>>>(width, height, d_lap, d_u);
	heatSolve<<<grid, block>>>(width, height, d_pixel, d_cmap, d_u, d_lap, 0.2f, origin, axes, time);
}

__global__ void waveSolve(int width, int height, uchar4 *d_pixel, uchar4 *d_cmap, float *d_u, float *d_ul, float *d_lap, float c, double2 origin, double2 axes, float time){
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int j=blockDim.y*blockIdx.y+threadIdx.y;
	if(i<width && j<height){
		int gid=j*width+i;
		float temp=d_u[gid];
		d_u[gid]=2*temp-d_ul[gid]+c*d_lap[gid];
		d_ul[gid]=temp;
		float x=fma((float)i/width, (float)axes.x, (float)origin.x);
		float y=fma((float)j/height, (float)axes.y, (float)origin.y);
		if(max(abs(x),abs(y))<1.E-2f){
			d_u[gid]=sinpif(time);
		}
		if(i==0 || j==0 || i==width-1 || j==height-1){
			int ii=i+(i==0)-(i==width-1);
			int jj=j+(j==0)-(j==height-1);
			d_u[gid]=lerp(d_ul[gid], d_ul[jj*width+ii],c);
		}
		int k=(int)(COLORDEPTH*(1+d_u[gid])/2);
		d_pixel[gid]=d_cmap[clamp(k, 0, COLORDEPTH-1)];
	}
}
void wavePDE(int width, int height, uchar4 *d_pixel, uchar4 *d_cmap, float *d_u, float *d_ul, float *d_lap, double2 origin, double2 axes, float time){
	static dim3 block(1,MAXTHREADS);
	static dim3 grid(ceil(width, block.x), ceil(height, block.y));
	laplacian<<<grid, block>>>(width, height, d_lap, d_u);
	waveSolve<<<grid, block>>>(width, height, d_pixel, d_cmap, d_u, d_ul, d_lap, 0.45f, origin, axes, time);
}


__device__ int interference(float x, float y, float t){
	float r1=sqrtf((x-1)*(x-1)+y*y);
	float r2=sqrtf((x+1)*(x+1)+y*y);
	float z1=expf(-r1*r1/4)*sinpif(5*r1-t);
	float z2=expf(-r2*r2/4)*sinpif(5*r2-t);
	return (int)(COLORDEPTH*(1.4351+z1+z2)/2.98489);
}
__device__ int mandelbrotf(float x, float y){
	cuComplex c=make_cuComplex(x, y);
	cuComplex z=make_cuComplex(x, y);
	int k=0;
	float z2=0.0;
	while(z2<4.f && k<COLORDEPTH){
		z=cuCaddf(cuCmulf(z, z), c);
		z2=z.x*z.x+z.y*z.y;
		k++;
	}
	return k-1;
}
__device__ int mandelbrotd(double x, double y){
	cuDoubleComplex c=make_cuDoubleComplex(x, y);
	cuDoubleComplex z=make_cuDoubleComplex(x, y);
	int k=0;
	double z2=0.0;
	while(z2<4.f && k<COLORDEPTH){
		z=cuCadd(cuCmul(z, z), c);
		z2=z.x*z.x+z.y*z.y;
		k++;
	}
	return k-1;
}

__global__ void kernelf(int width, int height, uchar4* d_pixel, uchar4 *d_cmap, double2 origin, double2 axes){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<width && j<height){
		int gid=j*width+i;
		float x=fma((float)i/width, (float)axes.x, (float)origin.x);
		float y=fma((float)j/height, (float)axes.y, (float)origin.y);
		int k=mandelbrotf(x, y);
		d_pixel[gid]=d_cmap[clamp(k,0,COLORDEPTH-1)];
	}
}
__global__ void kerneld(int width, int height, uchar4* d_pixel, uchar4 *d_cmap, double2 origin, double2 axes){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i<width && j<height){
		int k, gid=j*width+i;
		double x=(double)i/width*axes.x+origin.x;
		double y=(double)j/height*axes.y+origin.y;
		k=mandelbrotd(x, y);
		d_pixel[gid]=d_cmap[clamp(k,0,COLORDEPTH-1)];
	}
}



void init_kernel(int width, int height){
	int n=width*height;
	checkCudaErrors(cudaMalloc((void**)&d_u, n*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_ul, n*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_lap, n*sizeof(float)));
	checkCudaErrors(cudaMemset(d_u, 0.f, n*sizeof(float)));
	checkCudaErrors(cudaMemset(d_ul, 0.f, n*sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&d_cmap, COLORDEPTH*sizeof(uchar4)));
	jet<<<1, COLORDEPTH>>>(d_cmap, COLORDEPTH);
}

void launch_kernel(int width, int height, uchar4* d_pixel){
	static const dim3 block(MAXTHREADS);
	static const dim3 grid(ceil(width, block.x), ceil(height, block.y));
	static int nframes=0;
	float time=nframes/100.0;
	//wavePDE(width, height, d_pixel, d_cmap, d_u, d_ul, d_lap, origin, axes, time);
	kerneld<<<grid, block>>>(width, height, d_pixel, d_cmap, origin, axes);
	cudaThreadSynchronize();
	checkCudaErrors(cudaGetLastError());
	nframes++;
}
