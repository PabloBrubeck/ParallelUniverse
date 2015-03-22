// simpleVBO.cpp based on Rob Farber's code from drdobbs.com

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>
 
float animTime = 0.f;
 
// constants
const dim3 mesh(5, 5, 1);
const unsigned int RestartIndex = 0xffffffff;

struct mappedBuffer_t{
  GLuint vbo;
  GLuint typeSize;
  struct cudaGraphicsResource *cudaResource;
};

void launch_kernel(float4* d_pos, float4* d_norm, uchar4* d_color, dim3 mesh, float time);
 
// vbo variables
mappedBuffer_t vertexVBO = {NULL, sizeof(float4), NULL};
mappedBuffer_t normalVBO = {NULL, sizeof(float4), NULL};
mappedBuffer_t colorVBO  = {NULL, sizeof(uchar4), NULL};
 
// Create VBO
void createVBO(mappedBuffer_t* mbuf){
  // create buffer object
  glGenBuffers(1, &(mbuf->vbo) );
  glBindBuffer(GL_ARRAY_BUFFER, mbuf->vbo);
   
  // initialize buffer object
  unsigned int size=mesh.x*mesh.y*mesh.z*(mbuf->typeSize);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
   
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  // register buffer object with CUDA
  cudaGraphicsGLRegisterBuffer(&(mbuf->cudaResource), mbuf->vbo, cudaGraphicsMapFlagsNone);
}
 
// Delete VBO
void deleteVBO(mappedBuffer_t* mbuf){
	glBindBuffer(1, mbuf->vbo );
	glDeleteBuffers(1, &(mbuf->vbo) );
	cudaGraphicsUnregisterResource( mbuf->cudaResource );
	mbuf->cudaResource = NULL;
	mbuf->vbo = NULL;
}
 
void cleanupCuda(){
	deleteVBO(&vertexVBO);
	deleteVBO(&normalVBO);
	deleteVBO(&colorVBO);
	cudaDeviceReset();
}

// Run the Cuda part of the computation
void runCuda(){
	// map OpenGL buffer object for writing from CUDA
	float4 *d_pos;
	float4 *d_norm;
	uchar4 *d_color;
	size_t start;

	cudaGraphicsMapResources(1, &vertexVBO.cudaResource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&d_pos, &start, vertexVBO.cudaResource);
	cudaGraphicsMapResources(1, &normalVBO.cudaResource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&d_norm, &start, normalVBO.cudaResource);
	cudaGraphicsMapResources(1, &colorVBO.cudaResource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&d_color, &start, colorVBO.cudaResource);
 
    // execute the kernel
    launch_kernel(d_pos, d_norm, d_color, mesh, animTime);
 
    // unmap buffer object
	cudaGraphicsUnmapResources(1, &vertexVBO.cudaResource, NULL);
	cudaGraphicsUnmapResources(1, &normalVBO.cudaResource, NULL);
    cudaGraphicsUnmapResources(1, &colorVBO.cudaResource, NULL);
}
 
void initCuda(int argc, char** argv){
	// First initialize OpenGL context, so we can properly set the GL
	// for CUDA.  NVIDIA notes this is necessary in order to achieve
	// optimal performance with OpenGL/CUDA interop.  use command-line
	// specified CUDA device, otherwise use device with highest Gflops/s
	cudaGLSetGLDevice(findCudaDevice(argc, (const char **)argv));
   
	createVBO(&vertexVBO);
	createVBO(&normalVBO);
	createVBO(&colorVBO);
	// make certain the VBO gets cleaned up on program exit
	atexit(cleanupCuda);

	runCuda();
}
 
void renderCuda(int drawMode){
	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO.vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, normalVBO.vbo);
	glNormalPointer(GL_FLOAT, 0, 0);
	glEnableClientState(GL_NORMAL_ARRAY);
   
	glBindBuffer(GL_ARRAY_BUFFER, colorVBO.vbo);
	glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);
	
	size_t n=mesh.x*mesh.y*mesh.z;
	switch(drawMode){
		default:
		case GL_POINTS:
			glDrawArrays(GL_POINTS, 0, n);
			break;

		case GL_LINE_STRIP:
			for(int i=0 ; i<n; i+=mesh.x){
				glDrawArrays(GL_LINE_STRIP, i, mesh.x);
			}
			break;

		case GL_LINE_LOOP:
			for(int i=0 ; i<n; i+=mesh.x){
				glDrawArrays(GL_LINE_LOOP, i, mesh.x);
			}
			break;

		case GL_TRIANGLE_FAN:{
			int size=5*(mesh.y-1)*(mesh.x-1);
			static GLuint* qIndices=NULL;
			if(qIndices==NULL){
				qIndices=(GLuint*)malloc(mesh.z*size*sizeof(GLint));
				int index=0;
				// allocate and assign trianglefan indicies 
				for(int k=0; k<mesh.z; k++){
					for(int j=1; j<mesh.y; j++){
						for(int i=1; i<mesh.x; i++){
							qIndices[index++]=(k*mesh.y+j)*mesh.x+i; 
							qIndices[index++]=(k*mesh.y+j)*mesh.x+i-1; 
							qIndices[index++]=(k*mesh.y+j-1)*mesh.x+i-1; 
							qIndices[index++]=(k*mesh.y+j-1)*mesh.x+i; 
							qIndices[index++]=RestartIndex;
						}
					}
				}
			}
			for(int i=0; i<n; i+=size){
				glPrimitiveRestartIndexNV(RestartIndex);
				glEnableClientState(GL_PRIMITIVE_RESTART_NV);
				glDrawElements(GL_TRIANGLE_FAN, size, GL_UNSIGNED_INT, qIndices+i);
				glDisableClientState(GL_PRIMITIVE_RESTART_NV);
			}
			break;
		}
	}
 
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}