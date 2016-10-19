 // callbacksVBO.cpp adapted from Rob Farber's code from drdobbs.com

#include <cutil_math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <stdlib.h>


// constants
const dim3 mesh(1<<10, 1<<10);

struct mappedBuffer_t{
  GLuint vbo;
  GLuint typeSize;
  cudaGraphicsResource *cudaResource;
};

void launch_kernel(float4* d_pos, float4* d_norm, uchar4* d_color, uint4* d_index, dim3 mesh, float time);

// VBO variables
mappedBuffer_t vertexVBO = {0, sizeof(float4), NULL};
mappedBuffer_t normalVBO = {0, sizeof(float4), NULL};
mappedBuffer_t colorVBO  = {0, sizeof(uchar4), NULL};
mappedBuffer_t indexVBO  = {0, sizeof(uint4), NULL};

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
bool trackballMove = false;
int mouseButtons = 0;
int2 window, mouseStart, mouseEnd;

float3 lastPos;
float3 trans={0.f, 0.f, -5.f};
float4 axis={0.f, 0.f, 0.f, 1.f};
float4 quat={0.f, 0.f, 0.f, 1.f};
float *m=new float[16];

inline float4 quatMult(float4 a, float4 b){
	return make_float4(
		a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
		a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
		a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
		a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}

void trackball(int x, int y, int width, int height, float3 &v){
	v.x=(2.f*x-width)/width;
	v.y=(height-2.f*y)/height;
	float r=v.x*v.x+v.y*v.y;
	v.z=r<1? sqrtf(1-r): 0;
	v=normalize(v);
}
void updateMatrix(){
	m[0]=m[5]=m[10]=m[15]=1.f;
	m[3]=m[7]=m[11]=0.f;
	m[12]=trans.x;
	m[13]=trans.y;
	m[14]=trans.z;

	float temp=2.f*quat.x;
	float xx=temp*quat.x;
	float xw=temp*quat.w;
	m[1]=m[4]=temp*quat.y;
	m[2]=m[8]=temp*quat.z;
	temp=2.f*quat.y;
	float yy=temp*quat.y;
	float yw=temp*quat.w;
	m[6]=m[9]=temp*quat.z;
	temp=2.f*quat.z;
	float zz=temp*quat.z;
	float zw=temp*quat.w;

	m[0]-=yy+zz;
	m[5]-=xx+zz;
	m[10]-=xx+yy;
	m[1]+=zw; m[4]-=zw;
	m[2]-=yw; m[8]+=yw;
	m[6]+=xw; m[9]-=xw;
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(m);
}


// Callbacks for GLUT
void display(){
	// run CUDA kernel to generate vertex positions
	runCuda();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
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
	updateMatrix();
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
		if(state==GLUT_DOWN){
			trackball(x, y, window.x, window.y, lastPos);
			mouseStart=make_int2(x, y);
			trackballMove=false;
		}else{
			trackballMove=(mouseStart.x!=x || mouseStart.y!=y);
		}
	}
	mouseEnd=make_int2(x,y);
}
void mouseMotion(int x, int y){
	if(mouseButtons&1){
		static float3 curPos, n;
		trackball(x, y, window.x, window.y, curPos);
		n=cross(lastPos, curPos);
		lastPos=curPos;

		float w=sqrtf((1.f+sqrtf(1.f-dot(n,n)))/2.f);
		axis=make_float4(n/(2.f*w), w);
		quat=normalize(quatMult(axis, quat));
		updateMatrix();
	}
	if(mouseButtons&4){
		float dx=(x-mouseEnd.x)/(float)window.x;
		float dy=(y-mouseEnd.y)/(float)window.y;
		trans.z*=1.f+dy;
		updateMatrix();
	}
	mouseEnd=make_int2(x,y);
	glutPostRedisplay();
}

void timerEvent(int value){
	glutPostRedisplay();
	glutTimerFunc(10, timerEvent, 0);
}
void idle(){
	if(trackballMove){
		quat=normalize(quatMult(axis, quat));
		updateMatrix();
		glutPostRedisplay();
	}
}
