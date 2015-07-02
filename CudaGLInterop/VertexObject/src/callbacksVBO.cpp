 // callbacksVBO.cpp adapted from Rob Farber's code from drdobbs.com

#include <cutil_math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <stdlib.h>
#include <stdio.h>


// The user must create the following routines:
extern void runCuda();
extern void renderCuda(int);

// keyboard controls
int drawMode=GL_QUADS;
unsigned long pressed=0u;

void recordKey(unsigned char key, int a, int b, int c){
	if(key>=a && key<=b){ pressed |= 1<<(key-a+c); }
}
void deleteKey(unsigned char key, int a, int b, int c){
	if(key>=a && key<=b){ pressed &= ~(1<<(key-a+c)); }
}



// mouse controls
bool trackingMouse = false;
bool trackballMove = false;
bool redrawContinue = false;

int mouseButtons = 0;
int2 mouseStart, window;

float m[16];
float angle = 0.f;
float3 axis, lastPos;
float3 trans=make_float3(0.f, 0.f, -5.f);

void trackball(int x, int y, int width, int height, float3 &v){
	v.x=(2.f*x-width)/width;
	v.y=(height-2.f*y)/height;
	float r=v.x*v.x+v.y*v.y;
	v.z=r<1? sqrtf(1-r): 0;
	v=normalize(v);
}
void startMotion(int x, int y){
	trackingMouse=true;
	redrawContinue=false;
	mouseStart=make_int2(x, y);
	trackball(x, y, window.x, window.y, lastPos);
	trackballMove=true;
}
void stopMotion(int x, int y){
	trackingMouse=false;
	if(mouseStart.x!=x || mouseStart.y!=y){
		angle/=4.f;
		redrawContinue=true;
	}else{
		angle=0.f;
		redrawContinue=false;
		trackballMove=false;
	}
}



// Callbacks for GLUT
void display(){
	// run CUDA kernel to generate vertex positions
	runCuda();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// transform view matrix
	glMatrixMode(GL_MODELVIEW);
	if(trackballMove){
		glGetFloatv(GL_MODELVIEW_MATRIX, m);
		glLoadIdentity();
		glTranslatef(trans.x, trans.y, trans.z);
		glRotatef(angle, axis.x, axis.y, axis.z);
		glTranslatef(-trans.x, -trans.y, -trans.z);
		glMultMatrixf(m);
	}
	
	// render the data
	glTranslatef(trans.x, trans.y, trans.z);
	renderCuda(drawMode);
	glTranslatef(-trans.x, -trans.y, -trans.z);

	glutSwapBuffers();
	glutReportErrors();
}
void reshape(int w, int h){
	window=make_int2(w, h);
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (float)w/(float)h, 0.01, 100.0);
}

void keyPressed(unsigned char key, int x, int y){
	recordKey(key, 48, 57, 0);
	recordKey(key, 65, 90, 10);
	recordKey(key, 97, 122, 10);
	if(key==27){
		exit(0);
	}if(pressed & 0x00000002){
		switch(drawMode){
		case GL_POINTS: drawMode=GL_LINE_LOOP; break;
		case GL_LINE_LOOP: drawMode=GL_QUADS;  break;
		default: drawMode=GL_POINTS;
		}
	}if(pressed & 0x00000004){
		static bool fill=false;
		glPolygonMode(GL_FRONT_AND_BACK, fill? GL_FILL: GL_LINE);
		fill=!fill;
	}if(pressed & 0x00000008){
		static bool blend=false;
		if(blend){ glEnable(GL_BLEND); }else{ glDisable(GL_BLEND); }
		blend=!blend;
	}if(pressed & 0x00000010){
		static bool shade=false;
		glShadeModel(shade? GL_SMOOTH: GL_FLAT);
		shade=!shade;
	}if(pressed & 0x00000020){
		static bool light=false;
		if(light){	glEnable(GL_LIGHTING);	glEnable(GL_LIGHT0);
		}else{		glDisable(GL_LIGHTING); glEnable(GL_LIGHT0); }
		light=!light;
	}
	glutPostRedisplay();
}
void keyReleased(unsigned char key, int x, int y){
	deleteKey(key, 48, 57, 0);
	deleteKey(key, 65, 90, 10);
	deleteKey(key, 97, 122, 10);
	glutPostRedisplay();
}

void mouseButton(int button, int state, int x, int y){
	if(state == GLUT_DOWN) {
		mouseButtons |= 1<<button;
	}else if(state == GLUT_UP) {
		mouseButtons &= ~(1<<button);
	}
	if(button == GLUT_LEFT_BUTTON){
		if(state == GLUT_DOWN) {
			startMotion(x,y);
		}else{
			stopMotion(x,y);
		}
	}
}
void mouseMotion(int x, int y){
	float3 curPos, delta;
	trackball(x, y, window.x, window.y, curPos);
	if(trackingMouse){
		delta=curPos-lastPos;
		if(delta.x || delta.y || delta.z){
			axis=cross(lastPos, curPos);
			angle=573.f*length(axis);
			lastPos=curPos;
		}
	}
	glutPostRedisplay();
}

void timerEvent(int value){
	glutPostRedisplay();
	glutTimerFunc(10, timerEvent, 0);
}
void idle(){
	if(redrawContinue){
		glutPostRedisplay();
	}
}
