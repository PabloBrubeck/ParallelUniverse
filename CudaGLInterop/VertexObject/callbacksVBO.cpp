// callbacksVBO.cpp based on Rob Farber's code from drdobbs.com

#include <GL/glew.h>
#include <rendercheck_gl.h>
#include <nvGlutManipulators.h>

extern nv::GlutExamine manipulator;
extern float animTime;

// The user must create the following routines:
void initCuda(int argc, char** argv);
void runCuda();
void renderCuda(int);

// the default draw mode
int drawMode=GL_POINTS; 

// Callbacks for GLUT
void display(){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
    manipulator.applyTransform();

	// run CUDA kernel to generate vertex positions
	runCuda();
	// render the data
	renderCuda(drawMode);
	
	glutSwapBuffers();
	glutReportErrors();
	animTime+=0.01f;
}
void reshape(int w, int h){
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (float)w/(float)h, 0.01, 100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	manipulator.reshape(w, h);
}
void keyboard(unsigned char key, int x, int y){
	switch(key){
	case(27) :
		exit(0);
		break;
	case 'w':
	case 'W':
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
void mouse(int button, int state, int x, int y){
	manipulator.mouse(button, state, x, y);
}
void motion(int x, int y){
	 manipulator.motion(x, y);
}
void idle(){
	manipulator.idle();
	glutPostRedisplay();
}