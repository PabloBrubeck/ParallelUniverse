//callbacksVBO.cpp (Rob Farber)

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil_math.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>
#include "arcball.h"


extern int2 window;
extern bool refresh; 
extern float animTime;
extern void moveFingerX(int, float);
extern void moveFingerZ(int, float);


// The user must create the following routines:
void initCuda(int argc, char** argv);
void runCuda();
void renderCuda(int);
 
int drawMode=GL_POINTS; // the default draw mode
 

static float aspect_ratio = 1.0f;

// scene parameters
float3 eye=   make_float3(0.f, 0.f, -20.f);
float3 centre=make_float3(0.f, 0.f, 0.f);
float3 up=    make_float3(0.f, 1.f, 0.f);
float3 model =make_float3(0.f, -0.5f, 0.f);

// mouse controls
int2 mouseOld;
int mouseButtons = 0;

inline float3 rotate_x(float3 v, float sin_ang, float cos_ang){
	return make_float3(
		v.x,
		(v.y * cos_ang) + (v.z * sin_ang),
		(v.z * cos_ang) - (v.y * sin_ang)
		);
}
inline float3 rotate_y(float3 v, float sin_ang, float cos_ang){
	return make_float3(
		(v.x * cos_ang) + (v.z * sin_ang),
		v.y,
		(v.z * cos_ang) - (v.x * sin_ang)
		);
}

void resetView(){
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glTranslatef(model.x, model.y, model.z);
	gluPerspective(60.f, aspect_ratio, 0.01f, 100.0f);
	gluLookAt(
		eye.x, eye.y, eye.z,
		centre.x, centre.y, centre.z,
		up.x, up.y, up.z);
	// set up the arcball using the current projection matrix
	arcball_setzoom(-10.f/length(eye), eye, up);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
}

// Callbacks for GLUT
void display(){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// and perform the arcball rotation around the eye
	// (also disable depth test so they become the background)
	glPushMatrix();
	glDisable(GL_DEPTH_TEST);
	glTranslatef(eye.x, eye.y, eye.z);
	arcball_rotate();
	glEnable(GL_DEPTH_TEST);
	glPopMatrix();
	// now render the regular scene under the arcball rotation about 0,0,0
	// (generally you would want to render everything here)
	arcball_rotate();


	// run CUDA kernel to generate vertex positions
	runCuda();

	// render the data
	renderCuda(drawMode);
	
	glutSwapBuffers();

	animTime+=0.01f;
}
void reshape(int width, int height){
	window=make_int2(width, height);
	aspect_ratio = (float)width/(float)height;
	glViewport(0, 0, width, height);
	resetView();
}
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
	case 'k': case 'K':
		moveFingerX(4, -5.f);
		break;
	case 'm': case 'M':
		moveFingerX(4, 5.f);
		break;
	case 'j': case 'J':
		moveFingerX(3, -5.f);
		break;
	case 'n': case 'N':
		moveFingerX(3, 5.f);
		break;
	case 'h': case 'H':
		moveFingerX(2, -5.f);
		break;
	case 'b': case 'B':
		moveFingerX(2, 5.f);
		break;
	case 'g': case 'G':
		moveFingerX(1, -5.f);
		break;
	case 'v': case 'V':
		moveFingerX(1, 5.f);
		break;
	case 'o': case 'O':
		moveFingerZ(4, -5.f);
		break;
	case '0':
		moveFingerZ(4, 5.f);
		break;
	case 'i': case 'I':
		moveFingerZ(3, -5.f);
		break;
	case '9':
		moveFingerZ(3, 5.f);
		break;
	case 'u': case 'U':
		moveFingerZ(2, -5.f);
		break;
	case '8':
		moveFingerZ(2, 5.f);
		break;
	case 'y': case 'Y':
		moveFingerZ(1, -5.f);
		break;
	case '7':
		moveFingerZ(1, 5.f);
		break;
	}
	refresh=true;
	glutPostRedisplay();
}
void mouse(int button, int state, int x, int y){
	if (state == GLUT_DOWN){
		mouseOld=make_int2(x, y);
		arcball_start(window.x-x, y);
		mouseButtons |= 1<<button;
	}else{
		mouseButtons = 0;
	}
}
void motion(int x, int y){
	float dx=x-mouseOld.x;
	float dy=y-mouseOld.y;
	mouseOld=make_int2(x, y);
	if(mouseButtons&1){
		arcball_move(window.x-x, y);
	}else if(mouseButtons&4){
		eye.z*=(1.f+dy/100.f);
		resetView();
	}
}
void idle(){
	glutPostRedisplay();
}