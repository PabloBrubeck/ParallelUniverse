// simpleVBO.cpp adapted from Rob Farber's code from drdobbs.com

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// constants
const dim3 mesh(512, 512, 1);

struct mappedBuffer_t{
  GLuint vbo;
  GLuint typeSize;
  cudaGraphicsResource *cudaResource;
};

void launch_kernel(float4* d_pos, float4* d_norm, uchar4* d_color, uint4* d_index, dim3 mesh, float time);
 
// vbo variables
mappedBuffer_t vertexVBO = {NULL, sizeof(float4), NULL};
mappedBuffer_t normalVBO = {NULL, sizeof(float4), NULL};
mappedBuffer_t colorVBO  = {NULL, sizeof(uchar4), NULL};
mappedBuffer_t indexVBO  = {NULL, sizeof(uint4), NULL};

// Create VBO
void createVBO(mappedBuffer_t* mbuf, GLenum mode){
	// create buffer object
	glGenBuffers(1, &(mbuf->vbo));
	glBindBuffer(mode, mbuf->vbo);

	// initialize buffer object
	unsigned int size=mesh.x*mesh.y*mesh.z*(mbuf->typeSize);
	glBufferData(mode, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(mode, 0);

	// register buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&(mbuf->cudaResource),
		mbuf->vbo, cudaGraphicsMapFlagsNone));
	SDK_CHECK_ERROR_GL();
}
 
// Delete VBO
void deleteVBO(mappedBuffer_t* mbuf){
	glBindBuffer(1, mbuf->vbo);
	glDeleteBuffers(1, &(mbuf->vbo));
	checkCudaErrors(cudaGraphicsUnregisterResource(mbuf->cudaResource));
	mbuf->cudaResource=NULL;
	mbuf->vbo=NULL;
}
 
void cleanupCuda(){
	deleteVBO(&vertexVBO);
	deleteVBO(&normalVBO);
	deleteVBO(&colorVBO);
	deleteVBO(&indexVBO);
	cudaDeviceReset();
}

// Run the Cuda part of the computation
void runCuda(){
	static float animTime = 0.f;
	
	// map OpenGL buffer object for writing from CUDA
	static float4 *d_pos  ;
	static float4 *d_norm ;
	static uchar4 *d_color;
	static uint4  *d_index;
	static size_t start;
	
	checkCudaErrors(cudaGraphicsMapResources(1, &vertexVBO.cudaResource));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_pos, &start, vertexVBO.cudaResource));
	checkCudaErrors(cudaGraphicsMapResources(1, &normalVBO.cudaResource));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_norm, &start, normalVBO.cudaResource));
	checkCudaErrors(cudaGraphicsMapResources(1, &colorVBO.cudaResource));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_color, &start, colorVBO.cudaResource));
	checkCudaErrors(cudaGraphicsMapResources(1, &indexVBO.cudaResource));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_index, &start, indexVBO.cudaResource));
	
    // execute the kernel
    launch_kernel(d_pos, d_norm, d_color, d_index, mesh, animTime);
	animTime+=0.01f;

    // unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, &vertexVBO.cudaResource));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &normalVBO.cudaResource));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &colorVBO.cudaResource));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &indexVBO.cudaResource));
}
 
void initCuda(int argc, char** argv){
	// First initialize OpenGL context, so we can properly set the GL
	// for CUDA.  NVIDIA notes this is necessary in order to achieve
	// optimal performance with OpenGL/CUDA interop.  use command-line
	// specified CUDA device, otherwise use device with highest Gflops/s
	checkCudaErrors(cudaGLSetGLDevice(findCudaDevice(argc, (const char **)argv)));
   
	createVBO(&vertexVBO, GL_ARRAY_BUFFER);
	createVBO(&normalVBO, GL_ARRAY_BUFFER);
	createVBO(&colorVBO,  GL_ARRAY_BUFFER);
	createVBO(&indexVBO,  GL_ELEMENT_ARRAY_BUFFER);

	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO.vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, normalVBO.vbo);
	glNormalPointer(GL_FLOAT, sizeof(float4), 0);
	glBindBuffer(GL_ARRAY_BUFFER, colorVBO.vbo);
	glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);

	// make certain the VBO gets cleaned up on program exit
	atexit(cleanupCuda);

	runCuda();
}
 
void renderCuda(int drawMode){
	glEnable(GL_COLOR_MATERIAL);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	int n=mesh.x*mesh.y*mesh.z;
	switch(drawMode){
		default:
		case GL_POINTS:{
			glDrawArrays(GL_POINTS, 0, n);
		}break;
		case GL_LINE_LOOP:{
			for(int i=0 ; i<n; i+=mesh.x){
				glDrawArrays(GL_LINE_LOOP, i, mesh.x);
			}
		}break;
		case GL_QUADS:{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVBO.vbo);
			glDrawElements(GL_QUADS, 4*n, GL_UNSIGNED_INT, (void*)0);
		}break;
	}

	glDisable(GL_COLOR_MATERIAL);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}