#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

struct mappedBuffer_t{
  GLuint vbo;
  GLuint typeSize;
  struct cudaGraphicsResource *cudaResource;
};

class Shape{

};