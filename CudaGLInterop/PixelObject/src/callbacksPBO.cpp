// callbacksPBO.cpp adapted from Rob Farber's code from drdobbs.com

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cutil_math.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

//external variables
extern GLuint pbo;
extern GLuint textureID;
extern uint2 image;
extern float2 axes;
extern float2 origin;

// variables for keyboard control
int animFlag = 1;
float animTime = 0.0f;
float animInc = 0.1f;

// variables for mouse control
int2 mousePos, window;

// The user must create the following routines:
void runCuda();

void zoom(int x, int y, float z){
	origin.x+=((1-z)*axes.x*x)/window.x;
	origin.y+=((1-z)*axes.y*y)/window.y;
	axes*=z;
}

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
	window=make_int2(w,h);
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
void keyPressed(unsigned char key, int x, int y){
	float2 point;
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
	case 'z':
		zoom(x, window.y-y, 0.920f);
		break;
	case 'x':
		zoom(x, window.y-y, 1.087f);
		break;
	}

	// indicate the display must be redrawn
	glutPostRedisplay();
}
void keyReleased(unsigned char key, int x, int y){

}
void mouseButton(int button, int state, int x, int y){
	mousePos.x=x;
	mousePos.y=y;
}
void mouseMotion(int x, int y){
	origin.x-=(x-mousePos.x)*axes.x/window.x;
	origin.y+=(y-mousePos.y)*axes.y/window.y;
	mousePos.x=x;
	mousePos.y=y;
}
void timerEvent(int value){
	glutPostRedisplay();
	glutTimerFunc(10, timerEvent, 0);
}
void idle(){
	glutPostRedisplay();
}
