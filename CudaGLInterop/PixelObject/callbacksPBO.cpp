//callbacksPBO.cpp (Rob Farber)

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

// variables for keyboard control
int animFlag = 1;
float animTime = 0.0f;
float animInc = 0.1f;

//external variables
extern GLuint pbo;
extern GLuint textureID;
extern unsigned int image_width;
extern unsigned int image_height;

// The user must create the following routines:
void runCuda();

void display(){
	// run CUDA kernel
	runCuda();

	// Create a texture from the buffer
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// bind texture from PBO
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Note: glTexSubImage2D will perform a format conversion if the
	// buffer is a different format from the texture. We created the
	// texture with format GL_RGBA8. In glTexSubImage2D we specified
	// GL_BGRA and GL_UNSIGNED_INT. This is a fast-path combination

	// Note: NULL indicates the data resides in device memory
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, 
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	// Draw a single Quad with texture coordinates for each vertex.
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
	glEnd();

	// Don't forget to swap the buffers!
	glutSwapBuffers();

	// if animFlag is true, then indicate the display needs to be redrawn
	if(animFlag){
		glutPostRedisplay();
		animTime+=animInc;
	}
}

// Keyboard events handler for GLUT
void keyboard(unsigned char key, int x, int y){
	switch(key){
	case(27) :
		exit(0);
		break;
	case 'a': // toggle animation
	case 'A':
		animFlag = (animFlag) ? 0 : 1;
		break;
	case '-': // decrease the time increment for the CUDA kernel
		animInc -= 0.01;
		break;
	case '+': // increase the time increment for the CUDA kernel
		animInc += 0.01;
		break;
	case 'r': // reset the time increment 
		animInc = 0.01;
		break;
	}

	// indicate the display must be redrawn
	glutPostRedisplay();
}

// No mouse event handlers defined
void mouse(int button, int state, int x, int y){
}

void motion(int x, int y){
}