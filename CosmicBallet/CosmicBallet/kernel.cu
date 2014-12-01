#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <dos.h>
#include <conio.h>

#include "book.h"
#include "gl_helper.h"

#define MAXTHREADS 512
#define WIDTH 512
#define HEIGHT 512

using namespace std;

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
__global__
void interact(unsigned int *mass, float *r, float *v, unsigned char *bitmap, const int n, float dt) {
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	if(i>=n || j>=n || i==j){
		return;
	}
	int x=2*i+0;
	int y=2*i+1;

	float dx=r[2*j+0]-r[x];
	float dy=r[2*j+1]-r[y];

	float p=dx*dx+dy*dy;
	float s=invsqrt(p);

	float a=-0*mass[j]/p;
	float dvx=a*dx*s*dt;
	float dvy=a*dy*s*dt;

	atomicAdd(&(v[x]), dvx);
	atomicAdd(&(v[y]), dvy);
}
__global__
void move(unsigned int *mass, float *r, float *v, unsigned char *bitmap, const int n, float dt){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=n){
		return;
	}
	int x=2*i+0;
	int y=2*i+1;
	
	r[x]+=v[x]*dt;
	r[y]+=v[y]*dt;

	unsigned int m=mass[i];
	int offset=WIDTH*(int)r[y]+(int)r[x];
	if(offset>0 && offset<WIDTH*HEIGHT){
		bitmap[4*offset+0]=m;
		bitmap[4*offset+1]=m;
		bitmap[4*offset+2]=m;
		bitmap[4*offset+3]=255;
	}
}

__global__
void velField(float *d_r, float *d_vel, size_t n){
	int gid=blockDim.x*blockIdx.x*threadIdx.x;
	if(gid>=n){
		return;
	}
	int x=2*gid+0;
	int y=2*gid+1;
	int s=invsqrt(d_r[x]*d_r[x]+d_r[y]*d_r[y]);
	d_vel[x]=0;
	d_vel[y]=0;
}

void setRand(float *d_in, size_t n, float m, float s){
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandGenerateNormal(generator, d_in, n, m, s);
	curandDestroyGenerator(generator);
}

struct CPUBitmap {
	unsigned int x, y, n;
	unsigned char *pixels, *d_bitmap;
	
    void *dataBlock;
    void (*bitmapExit)(void*);

    CPUBitmap(int width, int height, int stars) {
		x=width;
		y=height;
		n=stars;

		pixels=new unsigned char[image_size()];
		HANDLE_ERROR(cudaMalloc((void**)&d_bitmap, image_size()));
    }

    ~CPUBitmap() {
        delete [] pixels;
		HANDLE_ERROR(cudaFree(d_bitmap));
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

	static CPUBitmap** get_bitmap_ptr( void ) {
        static CPUBitmap *gBitmap;
        return &gBitmap;
    }
    static void Draw(void) {
        CPUBitmap* bitmap=*(get_bitmap_ptr());
		size_t size=bitmap->image_size()*sizeof(unsigned char);
		int k=bitmap->n;

		unsigned int *d_mass;
		float *d_r, *d_vel;
		HANDLE_ERROR(cudaMalloc((void**)&d_mass, k*sizeof(unsigned int)));
		HANDLE_ERROR(cudaMalloc((void**)&d_r, 2*k*sizeof(unsigned int)));
		HANDLE_ERROR(cudaMalloc((void**)&d_vel, 2*k*sizeof(unsigned int)));

		HANDLE_ERROR(cudaMemset(d_mass, 255u, k*sizeof(unsigned int)));
		setRand(d_r, 2*k, 256, 40);
		velField<<<(k+MAXTHREADS-1)/MAXTHREADS, MAXTHREADS>>>(d_r, d_vel, k);

		float dt=1;
		do{
			HANDLE_ERROR(cudaMemset(bitmap->d_bitmap, 0, size));
		
			dim3 block(16, 16);
			dim3 grid((k+block.x-1)/block.x, (k+block.y-1)/block.y);
			interact<<<grid, block>>>(d_mass, d_r, d_vel, bitmap->d_bitmap, k, dt);
			move<<<(k+511)/512, 512>>>(d_mass, d_r, d_vel, bitmap->d_bitmap, k, dt);
		
			HANDLE_ERROR(cudaMemcpy(bitmap->pixels, bitmap->d_bitmap, size, cudaMemcpyDeviceToHost));
			glDrawPixels(bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
			glFlush();
			
		}while(true);
    }
};

int main(){
    CPUBitmap bitmap(WIDTH, HEIGHT, 2000);                              
    bitmap.display_and_exit();
    return 0;
}