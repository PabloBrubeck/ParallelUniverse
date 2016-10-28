#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuMatlab.h"
#include "vertex.h"
#include "particle.h"


using namespace std;


int main(int argc, char **argv){
	//waveExample(1024);
	//auto f=[] __device__ (double x){return sinpi(x);};
	//poisson(f, -1, 1, 32);

	dim3 mesh(1<<6, 1<<5, 1<<8);
	vertex(argc, argv, mesh, particleShadder);

	printf("Program terminated.\n");
	return 0;
}
