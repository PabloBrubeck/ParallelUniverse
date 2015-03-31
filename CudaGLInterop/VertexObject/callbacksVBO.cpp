// callbacksVBO.cpp adapted from Rob Farber's code from drdobbs.com

#include <GL/glew.h>
#include <GL/freeglut.h>

// The user must create the following routines:
void initCuda(int argc, char** argv);
void runCuda();
void renderCuda(int);

int drawMode=GL_POINTS; // the default draw mode
unsigned long pressed=0u;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// Callbacks for GLUT
void display(){
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
	glutPostRedisplay(); // seems unnecesary
	glutReportErrors();
}
void reshape(int w, int h){
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (double)w/(double)h, 0.01, 100.0);
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
		exit(0);
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
	glutTimerFunc(10, timerEvent, value);
}
void idle(){
	glutPostRedisplay();
}