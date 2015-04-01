// callbacksVBO.cpp adapted from Rob Farber's code from drdobbs.com

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "camera.h"

// The user must create the following routines:
void initCuda(int argc, char** argv);
void runCuda();
void renderCuda(int);

int drawMode=GL_POINTS; // the default draw mode
unsigned long pressed=0u;

static Camera *sCamera = new Camera(PxVec3(0.f, 0.f, -5.f), PxVec3(0.f, 0.f, 1.f));

void startRender(const PxVec3& cameraEye, const PxVec3& cameraDir){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Setup camera
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (float)glutGet(GLUT_WINDOW_WIDTH)/(float)glutGet(GLUT_WINDOW_HEIGHT), 0.01, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(cameraEye.x, cameraEye.y, cameraEye.z, cameraEye.x + cameraDir.x, cameraEye.y + cameraDir.y, cameraEye.z + cameraDir.z, 0.0f, 1.0f, 0.0f); 
}


// Callbacks for GLUT
void display(){
	runCuda();
	startRender(sCamera->getEye(), sCamera->getDir());

	// render the data
	renderCuda(drawMode);
	
	glutSwapBuffers();
	glutReportErrors();
}
void reshape(int w, int h){
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (float)w/(float)h, 0.01, 100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void keyPressed(unsigned char key, int x, int y){
	if(key>=48 && key<=57){
		pressed|=(1<<(key-48));
	}
	if(key>=65 && key<=90){
		pressed|=(1<<(key-65+10));
	}
	if(key>=97 && key<=122){
		pressed|=(1<<(key-97+10));
	}
	switch(key){
	case 27:
		exit(EXIT_SUCCESS);
		break;
	case 'w':
	case 'W':
		switch(drawMode){
		case GL_POINTS: 
			drawMode=GL_LINE_LOOP;
			break;
		case GL_LINE_LOOP: 
			drawMode=GL_QUADS; 
			break;
		default: 
			drawMode=GL_POINTS;
		}
		break;
		glutPostRedisplay();
	}
}
void keyReleased(unsigned char key, int x, int y){
	if(key>=48 && key<=57){
		pressed&=!(1<<(key-48));
	}
	if(key>=65 && key<=90){
		pressed&=!(1<<(key-65+10));
	}
	if(key>=97 && key<=122){
		pressed&=!(1<<(key-97+10));
	}
	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y){
	sCamera->handleMouse(button, state, x, y);
}
void motion(int x, int y){
	sCamera->handleMotion(x, y);
}

void timerEvent(int value){
	glutPostRedisplay();
	glutTimerFunc(10, timerEvent, value);
}
void idle(){
	glutPostRedisplay();
}