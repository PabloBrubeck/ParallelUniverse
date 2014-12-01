#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

#include "book.h"
#include "gl_helper.h"

#define WIDTH 512
#define HEIGHT 512

__device__
static float G=1.0f;
__device__
static float e=1.0f;

int divideCeil(int num, int den){
	return (num+den-1)/den;
}

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

struct cuComplex {
    float re, im;
	__device__
	cuComplex(float a, float b) : re(a), im(b){}
	__device__
	cuComplex operator+(const cuComplex& z) {
        return cuComplex(re+z.re, im+z.im);
    }
	__device__
	cuComplex operator-(const cuComplex& z) {
        return cuComplex(re-z.re, im-z.im);
    }
	__device__
	cuComplex operator*(const cuComplex& z) {
        return cuComplex(re*z.re-im*z.im, im*z.re+ re*z.im);
    }
	__device__
	cuComplex operator*(const float x) {
        return cuComplex(re*x, im*x);
    }
	__device__
	cuComplex operator/(const float x) {
        return cuComplex(re/x, im/x);
    }
	__device__
	float magnitude2( void ) {
        return re*re+im*im;
    } 
};

__global__
void interact(unsigned char *d_bitmap, float *mass, cuComplex *d_pos, cuComplex *d_vel0, cuComplex *d_velf, const float dt, const int n) {
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i>=n || j>=n || i==j){
		return;
	}

	cuComplex r=d_pos[j]-d_pos[i];
	float rr=r.magnitude2();
	cuComplex dv=rr>0.25? r*invsqrt(rr)*(G*mass[j]/rr)*dt: (d_vel0[j]-d_vel0[i])*((1+e)*mass[j]/(mass[i]+mass[j]));
	atomicAdd(&(d_velf[i].re), dv.re);
	atomicAdd(&(d_velf[i].im), dv.im);
}
__global__
void move(unsigned char *d_bitmap, float *mass, cuComplex *d_pos, cuComplex *d_vel, const float dt, const int n) {
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=n){
		return;
	}
	d_pos[i]=d_pos[i]+d_vel[i]*dt;
	int x=(int)d_pos[i].re;
	int y=(int)d_pos[i].im;
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
void setParams(float* d_mass, cuComplex *d_pos, cuComplex *d_vel, const int n){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		d_mass[i]=1;
		cuComplex r=d_pos[i]-cuComplex(255,255);
		float d=invsqrt(r.magnitude2());
		d_vel[i]=(cuComplex(-r.im, r.re)*d)/invsqrt(G*(n-1)*d);
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
    static CPUBitmap** get_bitmap_ptr(void) {
        static CPUBitmap *gBitmap;
        return &gBitmap;
    }

    
    // static method used for glut callbacks
    static void Draw(void){
		CPUBitmap* bitmap=*(get_bitmap_ptr());
		size_t size=bitmap->image_size();

		int n=1000;
		float dt=1;
		unsigned char *d_bitmap;
		float *d_mass;
		cuComplex *d_pos, *d_vel0, *d_velf;

		HANDLE_ERROR(cudaMalloc((void**)&d_bitmap, size));
		HANDLE_ERROR(cudaMalloc((void**)&d_mass, n*sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_pos, n*sizeof(cuComplex)));
		HANDLE_ERROR(cudaMalloc((void**)&d_vel0, n*sizeof(cuComplex)));
		HANDLE_ERROR(cudaMalloc((void**)&d_velf, n*sizeof(cuComplex)));

		int block1D=512;
		int grid1D=divideCeil(n, block1D);
		dim3 block2D(16, 32);
		dim3 grid2D(divideCeil(n, block2D.x), divideCeil(n, block2D.y));
		
		randset((float*)d_pos, 2*n, 256, 16);
		setParams<<<grid1D, block1D>>>(d_mass, d_pos, d_velf, n);
		
		do{
			HANDLE_ERROR(cudaMemset(d_bitmap, 0, size));
			HANDLE_ERROR(cudaMemcpy(d_vel0, d_velf, n*sizeof(cuComplex), cudaMemcpyDeviceToDevice));
			interact<<<grid2D, block2D>>>(d_bitmap, d_mass, d_pos, d_vel0, d_velf, dt, n);
			move<<<grid1D, block1D>>>(d_bitmap, d_mass, d_pos, d_velf, dt, n);
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