/*
 * vertex.h
 *
 *  Created on: Oct 18, 2016
 *      Author: pbrubeck
 */

#ifndef VERTEX_H_
#define VERTEX_H_

#include <stdlib.h>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_timer.h>

dim3 mesh;
void (*render)(dim3, float4*, float4*, uchar4*, uint4*);

struct mappedBuffer_t{
  GLuint vbo;
  GLuint typeSize;
  cudaGraphicsResource *cudaResource;
};

// VBO variables
mappedBuffer_t vertexVBO = {0, sizeof(float4), NULL};
mappedBuffer_t normalVBO = {0, sizeof(float4), NULL};
mappedBuffer_t colorVBO  = {0, sizeof(uchar4), NULL};
mappedBuffer_t indexVBO  = {0, sizeof(uint4), NULL};

// Timer for FPS calculations
StopWatchInterface *timer = NULL;

// mouse controls
bool trackballMove = false;
int mouseButtons = 0;
int2 window, mouseStart, mouseEnd;

float3 lastPos;
float3 trans={0.f, 0.f, -5.f};
float4 axis={0.f, 0.f, 0.f, 1.f};
float4 quat={0.f, 0.f, 0.f, 1.f};
float *m=new float[16];


// keyboard controls
int drawMode=GL_QUADS;
unsigned long pressed=0u;

void recordKey(unsigned char key, int a, int b, int c){
	if(key>=a && key<=b){ pressed |= 1<<(key-a+c); }
}
void deleteKey(unsigned char key, int a, int b, int c){
	if(key>=a && key<=b){ pressed &= ~(1<<(key-a+c)); }
}


void runCuda(){
	// map OpenGL buffer object for writing from CUDA
	static float4 *d_pos  ;
	static float4 *d_norm ;
	static uchar4 *d_color;
	static uint4  *d_index;
	static size_t start;

	checkCudaErrors(cudaGraphicsMapResources(1, &vertexVBO.cudaResource));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_pos, &start, vertexVBO.cudaResource));
	checkCudaErrors(cudaGraphicsMapResources(1, &normalVBO.cudaResource));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_norm, &start, normalVBO.cudaResource));
	checkCudaErrors(cudaGraphicsMapResources(1, &colorVBO.cudaResource));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_color, &start, colorVBO.cudaResource));
	checkCudaErrors(cudaGraphicsMapResources(1, &indexVBO.cudaResource));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_index, &start, indexVBO.cudaResource));

    // execute the kernel
    render(mesh, d_pos, d_norm, d_color, d_index);

    // unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, &vertexVBO.cudaResource));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &normalVBO.cudaResource));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &colorVBO.cudaResource));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &indexVBO.cudaResource));
}
void renderCuda(int drawMode){
	glEnable(GL_COLOR_MATERIAL);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	int n=mesh.x*mesh.y*mesh.z;
	switch(drawMode){
		default:
		case GL_POINTS:{
			glDrawArrays(GL_POINTS, 0, n);
		}break;
		case GL_LINE_LOOP:{
			for(int i=0 ; i<n; i+=mesh.x){
				glDrawArrays(GL_LINE_LOOP, i, mesh.x);
			}
		}break;
		case GL_QUADS:{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVBO.vbo);
			glDrawElements(GL_QUADS, 4*n, GL_UNSIGNED_INT, (void*)0);
		}break;
	}

	glDisable(GL_COLOR_MATERIAL);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}


inline float4 quatMult(float4 a, float4 b){
	return make_float4(
		a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
		a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
		a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
		a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}
void trackball(int x, int y, int width, int height, float3 &v){
	int s=min(width, height);
	v.x=(2.f*x-width)/s;
	v.y=(height-2.f*y)/s;
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

// Callbacks
void display(){
	// run CUDA kernel to generate vertex positions
	runCuda();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// render the data
	renderCuda(drawMode);
	glutSwapBuffers();
	glutReportErrors();
}
void fpsDisplay(){
	static int fpsCount=0;
	static int fpsLimit=60;
	sdkStartTimer(&timer);
	display();
	sdkStopTimer(&timer);
	if(++fpsCount==fpsLimit){
		float ifps=1000.f/sdkGetAverageTimerValue(&timer);
		char fps[256];
		sprintf(fps, "Cuda GL Interop Wrapper: %3.1f fps ", ifps);
		glutSetWindowTitle(fps);
		fpsCount=0;
		fpsLimit=(ifps<1)? 1:((ifps>200)? 200:(int)ifps);
		sdkResetTimer(&timer);
	}
}
void reshape(int w, int h){
	window=make_int2(w, h);
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (float)w/(float)h, 0.01, 200.0);
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

		float w=sqrtf((1.f+sqrtf(1.f-dot(n,n)))/2.f); // I was dividing by 2 before
		axis=make_float4(n/(2.f*w), w);
		quat=normalize(quatMult(axis, quat));
		updateMatrix();
	}
	if(mouseButtons&4){
		//float dx=(x-mouseEnd.x)/(float)window.x;
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

void createVBO(mappedBuffer_t* mbuf, GLenum mode){
	// create buffer object
	glGenBuffers(1, &(mbuf->vbo));
	glBindBuffer(mode, mbuf->vbo);

	// initialize buffer object
	unsigned int size=mesh.x*mesh.y*mesh.z*(mbuf->typeSize);
	glBufferData(mode, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(mode, 0);

	// register buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&(mbuf->cudaResource),
			mbuf->vbo, cudaGraphicsMapFlagsNone));
	SDK_CHECK_ERROR_GL();
}
void deleteVBO(mappedBuffer_t* mbuf){
	glBindBuffer(1, mbuf->vbo);
	glDeleteBuffers(1, &(mbuf->vbo));
	cudaGLUnregisterBufferObject(mbuf->vbo);
	mbuf->vbo=0;
}

void cleanupCuda(){
	deleteVBO(&vertexVBO);
	deleteVBO(&normalVBO);
	deleteVBO(&colorVBO);
	deleteVBO(&indexVBO);
	cudaDeviceReset();
}
void initCuda(int argc, char** argv){
	// First initialize OpenGL context, so we can properly set the GL
	// for CUDA.  NVIDIA notes this is necessary in order to achieve
	// optimal performance with OpenGL/CUDA interop.  Use command-line
	// specified CUDA device, otherwise use device with highest Gflops/s
	checkCudaErrors(cudaGLSetGLDevice(findCudaDevice(argc, (const char **)argv)));

	createVBO(&vertexVBO, GL_ARRAY_BUFFER);
	createVBO(&normalVBO, GL_ARRAY_BUFFER);
	createVBO(&colorVBO,  GL_ARRAY_BUFFER);
	createVBO(&indexVBO,  GL_ELEMENT_ARRAY_BUFFER);

	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO.vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, normalVBO.vbo);
	glNormalPointer(GL_FLOAT, sizeof(float4), 0);
	glBindBuffer(GL_ARRAY_BUFFER, colorVBO.vbo);
	glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0);

	// make certain the VBO gets cleaned up on program exit
	atexit(cleanupCuda);

	runCuda();
}
bool initGL(int* argc, char** argv){
	// create a window and GL context (also register callbacks)
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(720, 720);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Cuda GL Interop Wrapper (adapted from NVIDIA's simpleGL)");

	// register callbacks
	glutDisplayFunc(fpsDisplay);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyPressed);
	glutKeyboardUpFunc(keyReleased);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMotion);
	glutTimerFunc(10, timerEvent, 0);
	glutIdleFunc(idle);

	// initialize necessary OpenGL extensions
	glewInit();
	if(!glewIsSupported("GL_VERSION_2_0 ")){
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// Setup lighting
	float difuse[4]   = { 1.f, 1.f, 1.f, 1.f };
	float specular[4] =	{ 1.f, 1.f, 1.f, 1.f };
	float lightsrc[4] = { 0.f, 1.f, 1.f, 1.f };

	glClearColor(0.f, 0.f, 0.f, 0.f);

	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_CULL_FACE);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, difuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100.f);
	glLightfv(GL_LIGHT0, GL_POSITION, lightsrc);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	SDK_CHECK_ERROR_GL();
	return true;
}

void vertex(int argc, char** argv, dim3 msh, void (*fun)(dim3, float4*, float4*, uchar4*, uint4*)){
	mesh=msh;
	render=fun;
	sdkCreateTimer(&timer);

	if(!initGL(&argc, argv)){
		exit(EXIT_FAILURE);
	}

	initCuda(argc, argv);
	SDK_CHECK_ERROR_GL();

	// start rendering main loop
	glutMainLoop();

	// clean up
	sdkDeleteTimer(&timer);
	cudaThreadExit();
	exit(EXIT_SUCCESS);
}


#endif /* VERTEX_H_ */
