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
int2 mouseStart, mouseEnd, window;

float3 lastPos;
float3 trans={0.f, 0.f, -5.f};
float4 axis, quat={0.f, 0.f, 0.f, 1.f};

float4 quatMult(float4 a, float4 b){
	float x=a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
	float y=a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
	float z=a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
	float w=a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
	return make_float4(x, y, z, w);
}

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
		redrawContinue=true;
	}else{
		axis.w=1.f;
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
	if(trackballMove && length(axis)>0){
		quat=normalize(quatMult(axis, quat));
	}
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(trans.x, trans.y, trans.z);
	glRotatef(114.592f*acosf(quat.w), quat.x, quat.y, quat.z);
	
	// render the data
	renderCuda(drawMode);

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
	mouseEnd=make_int2(x,y);
}
void mouseMotion(int x, int y){
	float3 curPos;
	trackball(x, y, window.x, window.y, curPos);
	if(trackingMouse){
		float3 n=cross(lastPos, curPos);
		float n2=dot(n, n);
		if(n2 > 0.f){
			float ch=sqrtf((1.f+sqrtf(1.f-n2))/2.f);
			axis=make_float4(n/(2.f*ch), ch);
			lastPos=curPos;
		}
	}
	if(mouseButtons&4){
		trans.z*=1+(y-mouseEnd.y)/(float)window.y;
	}
	mouseEnd=make_int2(x,y);
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
