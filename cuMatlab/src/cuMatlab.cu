#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuMatlab.h"
#include "vertex.h"
#include "geometry.h"
#include "particle.h"


using namespace std;


int main(int argc, char **argv){
	//waveExample(1024);
	//auto f=[] __device__ (double x){return sinpi(x);};
	//poisson(f, -1, 1, 32);

	dim3 mesh(1<<4, 1<<5, 1<<7);
	vertex(argc, argv, mesh, particleExample);

	printf("Program terminated.\n");
	return 0;
}
