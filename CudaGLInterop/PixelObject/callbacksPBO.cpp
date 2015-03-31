// callbacksPBO.cpp adapted from Rob Farber's code from drdobbs.com

#include <GL/glew.h>
#include <GL/freeglut.h>
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
extern uint2 image;

// The user must create the following routines:
void runCuda();

// Callbacks for GLUT
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
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.x, image.y, 
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
	glutReportErrors();

	// if animFlag is true, then indicate the display needs to be redrawn
	if(animFlag){
		glutPostRedisplay();
		animTime+=animInc;
	}
}
void reshape(int w, int h){
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
void keyPressed(unsigned char key, int x, int y){
	switch(key){
	case(27) :
		exit(0);
		break;
	case 'a': // toggle animation
	case 'A':
		animFlag = (animFlag) ? 0 : 1;
		break;
	case '-': // decrease the time increment for the CUDA kernel
		animInc -= 0.01f;
		break;
	case '+': // increase the time increment for the CUDA kernel
		animInc += 0.01f;
		break;
	case 'r': // reset the time increment 
		animInc = 0.01f;
		break;
	}

	// indicate the display must be redrawn
	glutPostRedisplay();
}
void keyReleased(unsigned char key, int x, int y){
}
void mouse(int button, int state, int x, int y){
}
void motion(int x, int y){
}
void timerEvent(int value){
	glutPostRedisplay();
	glutTimerFunc(10, timerEvent, 0);
}
void idle(){
	glutPostRedisplay();
}