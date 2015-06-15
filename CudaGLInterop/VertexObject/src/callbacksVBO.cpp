 // callbacksVBO.cpp adapted from Rob Farber's code from drdobbs.com

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <stdlib.h>
#include <stdio.h>

// The user must create the following routines:
void runCuda();
void renderCuda(int);

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// keyboard controls
int drawMode=GL_QUADS;

unsigned long pressed=0u;
void recordKey(unsigned char key, int a, int b, int c){
	if(key>=a && key<=b){ pressed|=(1<<(key-a+c)); }
}
void deleteKey(unsigned char key, int a, int b, int c){
	if(key>=a && key<=b){ pressed-=(1<<(key-a+c)); }
}

// Callbacks for GLUT
void display(){
	// run CUDA kernel to generate vertex positions
	runCuda();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

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

void mouse(int button, int state, int x, int y){
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1<<button;
	} else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}
	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}
void motion(int x, int y){
	float dx, dy;
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;
	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	} else if (mouse_buttons & 4) {
		translate_z += dy * 0.01f;
	}
	mouse_old_x = x;
	mouse_old_y = y;
}

void timerEvent(int value){
	glutPostRedisplay();
	glutTimerFunc(10, timerEvent, 0);
}
void idle(){
	glutPostRedisplay();
}
