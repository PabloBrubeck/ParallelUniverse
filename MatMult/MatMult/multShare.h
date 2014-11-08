#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <algorithm>
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	float* elements;
	int stride;
} Matrix;
// Thread block size
#define BLOCK_SIZE 16
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);