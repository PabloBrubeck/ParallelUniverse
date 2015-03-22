//callbacksVBO.cpp (Rob Farber)

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil_math.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

extern uint2 window; 
extern float animTime;
extern void moveFingerX(int a, float d);
extern void moveFingerZ(int a, float d);
extern bool refresh;

// The user must create the following routines:
void initCuda(int argc, char** argv);
void runCuda();
void renderCuda(int);
 
int drawMode=GL_POINTS; // the default draw mode
 
// mouse controls
int2 mouseOld;
int mouseButtons = 0;
float zoom = 1.f;
float4 quat={0.f, 0.f, 1.f, 0.f};

void getArcball(float3 &arcball, int i, int j){
	float x=((float)(2*i))/window.x-1.f;
	float y=((float)(2*j))/window.y-1.f;
	float z=0.f;
	float r2=x*x+y*y;
	if(r2<=1){
		z=sqrt(1-r2);
	}else{
		float r=sqrt(r2);
		x/=r;
		y/=r;
	}
	arcball=make_float3(x, -y, z);
}

// Display callback for GLUT
void display(){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glRotatef(quat.w, quat.x, quat.y, quat.z);
	quat.w=0;
	if(zoom!=1.f){
		glScalef(zoom, zoom, zoom);
		zoom=1.f;
	}

	// run CUDA kernel to generate vertex positions
	runCuda();

	// render the data
	renderCuda(drawMode);
	
	glutSwapBuffers();
	glutPostRedisplay();

	animTime+=0.01f;
}
 
// Events handlers for GLUT
void reshape(int x, int y){
	window=make_uint2(x, y);
	glViewport(0, 0, x, y);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)x/(GLfloat)y, 0.01, 100.0);
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
	if(state==GLUT_DOWN){
		mouseButtons |= 1<<button;
		mouseOld=make_int2(x, y);
	}else if(state == GLUT_UP){
		mouseButtons = 0;
	}
	glutPostRedisplay();
}
void motion(int x, int y){
	static float3 arcNew, arcOld;
	float dx=x-mouseOld.x;
	float dy=y-mouseOld.y;
	
	if(mouseButtons&1){
		getArcball(arcNew, x, y);
		getArcball(arcOld, mouseOld.x, mouseOld.y);
		quat=make_float4(cross(arcOld, arcNew), 
			57.3f*acosf(min(1.f, dot(arcOld, arcNew))));
	}else if(mouseButtons&4){
		zoom=(1.f+dy/100.f);
	}
	mouseOld=make_int2(x, y);
}