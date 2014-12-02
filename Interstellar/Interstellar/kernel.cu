#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

#include "book.h"
#include "gl_helper.h"

#define MAXTHREADS 512u
#define WIDTH 512
#define HEIGHT 512

int divideCeil(int num, int den){
	return (num+den-1)/den;
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

__device__
static float G=0.1f;
__device__
float invsqrt(float x){
	long i;
	float x2, y;
	const float threehalfs = 1.5F;
	x2=x*0.5F;
	y=x;
	i=*(long*)&y;                // evil floating point bit level hacking
	i=0x5f3759df-(i>>1);         // what the fuck?
	y=*(float*)&i;
	y=y*(threehalfs-(x2*y*y));   // 1st iteration
    y=y*(threehalfs-(x2*y*y));   // 2nd iteration, this can be removed
	return y;
}
__device__
float3 operator+(const float3& u, const float3& v) {
    return make_float3(u.x+v.x, u.y+v.y, u.z+v.z);
}
__device__
float3 operator-(const float3& u, const float3& v) {
    return make_float3(u.x-v.x, u.y-v.y, u.z-v.z);
}
__device__
float3 operator*(const float3& u, const float d) {
    return make_float3(u.x*d, u.y*d, u.z*d);
}
__device__
float3 operator/(const float3& u, const float d) {
    return make_float3(u.x/d, u.y/d, u.z/d);
}
__device__
float magnitude2(const float3& v) {
    return v.x*v.x+v.y*v.y+v.z*v.z;
}

__global__
void mapMagnitude2(float3 *d_vec, float* d_mag, const int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_mag[i]=magnitude2(d_vec[i]);
	}
}
__global__
void reduceMax(float *d_in, float *d_out, const size_t elements)
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
float getMax(float *d_in, int n){
	int grid, block=MAXTHREADS;
	float *h_out=new float();
	do{
		grid=(n+block-1)/block;
		if(grid==1){
			block=nextPowerOf2(n);
		}
		reduceMax<<<grid, block, block*sizeof(float)>>>(d_in, d_in, n);
		n=grid;
	}while(grid>1);
	HANDLE_ERROR(cudaMemcpy(h_out, d_in, sizeof(float), cudaMemcpyDeviceToHost));
	return *h_out;
}

__global__
void interact(float *mass, float3 *d_pos, float3 *d_acc, const int n){
	extern __shared__ float3 temp[];
	int tid=threadIdx.x;
	int i=blockIdx.x;
	int j=blockIdx.y*blockDim.x+tid;
	if(j>=n || i==j){
		temp[tid]=make_float3(0.0f, 0.0f, 0.0f);
	}else{
		float3 r=d_pos[j]-d_pos[i];
		float rr=magnitude2(r);
		temp[tid]=r*invsqrt(rr)*(G*mass[j]/rr);
	}
    __syncthreads();
	//Reduction
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid<s){
            temp[tid]=temp[tid]+temp[tid+s];
        }
        __syncthreads();
    }
    if(tid==0){
		atomicAdd(&(d_acc[i].x), temp[0].x);
		atomicAdd(&(d_acc[i].y), temp[0].y);
		atomicAdd(&(d_acc[i].z), temp[0].z);
    }
}
__global__
void move(unsigned char *d_bitmap, float *mass, float3 *d_pos, float3 *d_vel, float3 *d_acc, float dt, const int n) {
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=n){
		return;
	}
	d_vel[i]=d_vel[i]+d_acc[i]*dt;
	d_pos[i]=d_pos[i]+d_vel[i]*dt;
	int x=(int)d_pos[i].x;
	int y=(int)d_pos[i].y;
	if(x>=0 && x<WIDTH && y>=0 && y<HEIGHT){
		unsigned int m=255;
		int offset=y*WIDTH+x;
		d_bitmap[4*offset+0]=m;
		d_bitmap[4*offset+1]=m;
		d_bitmap[4*offset+2]=m;
		d_bitmap[4*offset+3]=255;
	}
}

__global__
void setParams(float* d_mass, float3 *d_pos, float3 *d_vel, float3 *d_acc, const int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		float3 r=d_pos[i]-make_float3(256, 256, 256);
		float3 theta=make_float3(r.y, -r.x, 0);
		theta=theta*invsqrt(magnitude2(theta));
		//d_vel[i]=theta*invsqrt(2097152.0f/(G*n*magnitude2(r)));
		d_vel[i]=make_float3(0.0f, 0.0f, 0.0f);
		d_acc[i]=make_float3(0.0f, 0.0f, 0.0f);
		d_mass[i]=1;
	}
}
void randset(float* d_in, size_t n, float m, float s){
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandGenerateNormal(generator, d_in, n, m, s);
	curandDestroyGenerator(generator);
}

struct CPUBitmap {
	unsigned char *pixels;
    int x, y;

    void *dataBlock;
    void (*bitmapExit)(void*);

    CPUBitmap(int width, int height) {
		x=width;
        y=height;
		HANDLE_ERROR(cudaMallocHost((void**)&pixels, 4*width*height));
    }
    ~CPUBitmap() {
        delete[] pixels;
    }

    unsigned char* get_ptr( void ) const   { 
		return pixels; 
	}
    static CPUBitmap** get_bitmap_ptr(void) {
        static CPUBitmap *gBitmap;
        return &gBitmap;
    }
	long image_size( void ) const { 
		return 4*x*y; 
	}

	void display_and_exit(void(*e)(void*)=NULL){
        CPUBitmap** bitmap=get_bitmap_ptr();
        *bitmap=this;
        bitmapExit=e;
        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int c=1;
        char* dummy="";
        glutInit(&c, &dummy);
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
        glutInitWindowSize(x, y);
        glutCreateWindow("bitmap");
        glutDisplayFunc(Draw);
        glutMainLoop();
    }
    
    // static method used for glut callbacks
    static void Draw(void){
		CPUBitmap* bitmap=*(get_bitmap_ptr());
		size_t size=bitmap->image_size();

		int n=1024;
		float dt, dvmax=4.0f;
		unsigned char *d_bitmap;
		float *d_mass, *d_aux;
		float3 *d_pos, *d_vel, *d_acc;

		HANDLE_ERROR(cudaMalloc((void**)&d_bitmap, size));
		HANDLE_ERROR(cudaMalloc((void**)&d_aux, n*sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_mass, n*sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_pos, n*sizeof(float3)));
		HANDLE_ERROR(cudaMalloc((void**)&d_vel, n*sizeof(float3)));
		HANDLE_ERROR(cudaMalloc((void**)&d_acc, n*sizeof(float3)));

		int block1D=MAXTHREADS;
		int grid1D=divideCeil(n, block1D);
		dim3 block2D(MAXTHREADS);
		dim3 grid2D(n, divideCeil(n, MAXTHREADS));
		int bytes=MAXTHREADS*sizeof(float3);
		
		randset((float*)d_pos, 3*n, 256, 32);
		setParams<<<grid1D, block1D>>>(d_mass, d_pos, d_vel, d_acc, n);
		do{
			HANDLE_ERROR(cudaMemsetAsync(d_bitmap, 0, size));
			interact<<<grid2D, block2D, bytes>>>(d_mass, d_pos, d_acc, n);

			mapMagnitude2<<<grid1D, block1D>>>(d_acc, d_aux, n);
			dt=dvmax/sqrt(getMax(d_aux, n));
			cudaDeviceSynchronize();

			move<<<grid1D, block1D>>>(d_bitmap, d_mass, d_pos, d_vel, d_acc, dt, n);
			HANDLE_ERROR(cudaMemcpy(bitmap->pixels, d_bitmap, size, cudaMemcpyDeviceToHost));
			glDrawPixels(bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
			glFlush();
		}while(true);
    }
};

int main( void ) {
    CPUBitmap bitmap(WIDTH, HEIGHT);                              
    bitmap.display_and_exit();
}