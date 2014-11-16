#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <algorithm>

#include "book.h"
#include "gl_helper.h"

#define LENGTH 1024
#define HEIGHT 512
#define PI 3.1415926535898

struct cuComplex {
    double   r;
    double   i;
    __device__ cuComplex( double a, double b ) : r(a), i(b)  {}
    __device__ double magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& z) {
        return cuComplex(r*z.r - i*z.i, i*z.r + r*z.i);
    }
    __device__ cuComplex operator+(const cuComplex& z) {
        return cuComplex(r+z.r, i+z.i);
    }
};

__device__
int julia(double jx, double jy) {
	cuComplex c(jx, jy);
	cuComplex z(jx, jy);
	int i=0;
	do{
		z=z*z+c;
		i++;
	}while(z.magnitude2()<4 && i<258);
	return i;
}

__global__
void kernel(unsigned char *ptr, const double h, const double k, const double zoom) {
	// map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x>=LENGTH || y>=HEIGHT){
		return;
	}
	int offset=y*LENGTH+x;

	// now calculate the value at that position
	double range=min(HEIGHT,LENGTH)/2;
	double jx=zoom*(x-h-LENGTH/2)/range;
	double jy=zoom*(HEIGHT/2-y-k)/range;
	
	int i=julia(jx, jy);
	
	double r=max(0, 256-4*abs(i-192));
	double g=max(0, 256-4*abs(i-64));
	double b=max(0, 256-4*i);
	ptr[offset*4 + 0] = (int)(r*r/255);
	ptr[offset*4 + 1] = (int)(g*g/255);
	ptr[offset*4 + 2] = (int)(b*b/255);
	ptr[offset*4 + 3] = 255;
}

struct CPUBitmap {

	unsigned char *pixels;
	unsigned char *dev_bitmap;
    int x, y;
	int h, k;
	int htemp, ktemp;
	double zoom;

    void *dataBlock;
    void (*bitmapExit)(void*);

    CPUBitmap( int width, int height) {
		pixels = new unsigned char[width * height * 4];
		x=width;
        y=height;
		
		HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, image_size()));
		h=0;
		k=0;
		zoom=1.0f;
    }

    ~CPUBitmap() {
        delete [] pixels;
		HANDLE_ERROR(cudaFree(dev_bitmap));
    }

    unsigned char* get_ptr( void ) const   { 
		return pixels; 
	}
    long image_size( void ) const { 
		return x * y * 4; 
	}

    void display_and_exit( void(*e)(void*) = NULL ) {
        CPUBitmap**   bitmap = get_bitmap_ptr();
        *bitmap = this;
        bitmapExit = e;
        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int c=1;
        char* dummy = "";
        glutInit( &c, &dummy );
        glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
        glutInitWindowSize( x, y );
        glutCreateWindow( "bitmap" );
        glutKeyboardFunc(Key);
		glutMouseFunc(Mouse);
		glutMotionFunc(Motion);
        glutDisplayFunc(Draw);
        glutMainLoop();
    }

     // static method used for glut callbacks
    static CPUBitmap** get_bitmap_ptr( void ) {
        static CPUBitmap *gBitmap;
        return &gBitmap;
    }

    // static methods used for glut callbacks
    static void Key(unsigned char key, int x, int y) {
		CPUBitmap* bitmap = *(get_bitmap_ptr());
		int xm=x-LENGTH/2;
        int ym=y-HEIGHT/2;
		double scale=1.1f;
		switch (key) {
			case 'z':
				bitmap->zoom/=scale;
				bitmap->k=(int)((bitmap->k-ym)*scale+ym+0.5);
				bitmap->h=(int)((bitmap->h-xm)*scale+xm+0.5);
				Draw();
				break;
		    case 'x':
				bitmap->zoom*=scale;
				bitmap->k=(int)((bitmap->k-ym)/scale+ym+0.5);
				bitmap->h=(int)((bitmap->h-xm)/scale+xm+0.5);
				Draw();
				break;
            case 27:
                if (bitmap->dataBlock != NULL && bitmap->bitmapExit != NULL)
                    bitmap->bitmapExit( bitmap->dataBlock );
                exit(0);
        }
    }
	static void Mouse(int button, int state, int x, int y){
		if(state==GLUT_DOWN){
			CPUBitmap* bitmap=*(get_bitmap_ptr());
			bitmap->htemp=x;
			bitmap->ktemp=y;
		}
	}
	static void Motion(int x, int y){
		CPUBitmap* bitmap=*(get_bitmap_ptr());

		bitmap->h+=x-bitmap->htemp;
		bitmap->k+=y-bitmap->ktemp;

		bitmap->htemp=x;
		bitmap->ktemp=y;

		Draw();
	}

    // static method used for glut callbacks
    static void Draw(void) {
        CPUBitmap* bitmap=*(get_bitmap_ptr());
		size_t size=bitmap->image_size();

		dim3 blockSize(32, 16);
		dim3 gridSize((LENGTH+blockSize.x-1)/blockSize.x, (HEIGHT+blockSize.y-1)/blockSize.y);
		kernel<<<gridSize, blockSize>>>(bitmap->dev_bitmap, bitmap->h, bitmap->k, bitmap->zoom);
		HANDLE_ERROR(cudaMemcpy(bitmap->pixels, bitmap->dev_bitmap, size, cudaMemcpyDeviceToHost));
		
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);
        glFlush();
    }
};

int main( void ) {
    CPUBitmap bitmap(LENGTH, HEIGHT);                              
    bitmap.display_and_exit();
}