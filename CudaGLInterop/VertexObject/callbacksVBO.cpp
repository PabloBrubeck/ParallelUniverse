//callbacksVBO.cpp (Rob Farber)

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>
 
extern float animTime;
 
// The user must create the following routines:
void initCuda(int argc, char** argv);
void runCuda();
void renderCuda(int);
 
int drawMode=GL_LINE_LOOP; // the default draw mode
 
// mouse controls
int2 mouseOld;
int mouseButtons = 0;
float2 rotate=make_float2(0.f, 0.f);
float translate_z = -3.f;
double zoom=1.0;

//! Display callback for GLUT
void display(){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.f, 0.f, translate_z);
	glRotatef(rotate.x, 1.f, 0.f, 0.f);
	glRotatef(rotate.y, 0.f, 1.f, 0.f);
	
	// run CUDA kernel to generate vertex positions
	runCuda();

	// render the data
	renderCuda(drawMode);
	
	glutSwapBuffers();
	glutPostRedisplay();

	animTime+=0.01f;
}
 
//! Keyboard events handler for GLUT
void keyboard(unsigned char key, int x, int y){
	switch(key){
	case(27) :
		exit(0);
		break;
	case 'd':
	case 'D':
		switch(drawMode){
		case GL_POINTS: drawMode = GL_LINE_STRIP; break;
		case GL_LINE_STRIP: drawMode = GL_LINE_LOOP; break;
		case GL_LINE_LOOP: drawMode = GL_TRIANGLE_FAN; break;
		default: drawMode=GL_POINTS;
		}
		break;
	}
	glutPostRedisplay();
}

// Mouse event handlers for GLUT
void mouse(int button, int state, int x, int y){
	if(state==GLUT_DOWN){
		mouseButtons |= 1<<button;
	}else if(state == GLUT_UP){
		mouseButtons = 0;
	}

	if(mouseButtons&8){
		zoom*=0.9;
	}else if(mouseButtons&16){
		zoom*=1.1;
	}

	mouseOld=make_int2(x, y);
	glutPostRedisplay();
}

void motion(int x, int y){
	float dx=x-mouseOld.x;
	float dy=y-mouseOld.y;
	if(mouseButtons&1){
		rotate.x+=dy*0.2f;
		rotate.y+=dx*0.2f;
	}else if(mouseButtons&4){
		translate_z+=dy*0.01;
	}
	mouseOld=make_int2(x, y);
}