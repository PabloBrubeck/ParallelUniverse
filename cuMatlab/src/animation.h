/*
 * animation.h
 *
 *  Created on: Aug 3, 2016
 *      Author: pbrubeck
 */

#ifndef ANIMATION_H_
#define ANIMATION_H_

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_timer.h>
#include <rendercheck_gl.h>

GLuint pbo;
GLuint textureID;

// Timer for FPS calculations
StopWatchInterface *timer = NULL;

// variables for mouse control
int2 window, mousePos;
int2 poi, roi;

// variables for keyboard control
int animFlag = 1;
int animTime = 0;

double *axes=new double[4];
int width, height;
void (*render)(int, int, uchar4*, double, double, double, double);

void runCuda(){
	uchar4 *d_pixel=NULL;

	// map OpenGL buffer object for writing from CUDA on a single GPU
	// no data is moved (Win & Linux). When mapped to CUDA, OpenGL
	// should not use this buffer
	cudaGLMapBufferObject((void**)&d_pixel, pbo);

	// execute the kernel
	double xmin=axes[0];
	double ymin=axes[1];
	double xmax=xmin+axes[2];
	double ymax=ymin+axes[3];
	render(width, height, d_pixel, xmin, xmax, ymin, ymax);

	// unmap buffer object
	cudaGLUnmapBufferObject(pbo);
}

void zoom(int x, int y, double z){
	axes[0]+=axes[2]*(1-z)*(poi.x+(x*roi.x)/(double)window.x)/width;
	axes[1]+=axes[3]*(1-z)*(poi.y+(y*roi.y)/(double)window.y)/height;
	axes[2]*=z;
	axes[3]*=z;
}
void setROI(int x, int y, double z){
	poi.x+=((1-z)*(x*roi.x))/(double)window.x;
	poi.y+=((1-z)*(y*roi.y))/(double)window.y;
	roi.x=clamp((int)(z*roi.x), 1, width);
	roi.y=clamp((int)(z*roi.y), 1, height);
	if(poi.x<0){
		poi.x=0;
	}else if(poi.x+roi.x>width){
		poi.x=width-roi.x;
	}
	if(poi.y<0){
		poi.y=0;
	}else if(poi.y+roi.y>height){
		poi.y=height-roi.y;
	}
}

// Callbacks
void display(){
	// run CUDA kernel
	runCuda();

	// Create a texture from the buffer
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// bind texture from PBO
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, textureID);

	// Note: glTexSubImage2D will perform a format conversion if the
	// buffer is a different format from the texture. We created the
	// texture with format GL_RGBA8. In glTexSubImage2D we specified
	// GL_BGRA and GL_UNSIGNED_INT. This is a fast-path combination

	// Note: NULL indicates the data resides in device memory
	glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, width, height,
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	// Draw a single Quad with texture coordinates for each vertex.
	glBegin(GL_QUADS);
	glTexCoord2i(poi.x,       poi.y);		glVertex3f(0.0f, 0.0f, 0.0f);
	glTexCoord2i(poi.x+roi.x, poi.y);		glVertex3f(1.0f, 0.0f, 0.0f);
	glTexCoord2i(poi.x+roi.x, poi.y+roi.y);	glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2i(poi.x,       poi.y+roi.y);	glVertex3f(0.0f, 1.0f, 0.0f);
	glEnd();

	// Don't forget to swap the buffers!
	glutSwapBuffers();
	glutReportErrors();

	// if animFlag is true, then indicate the display needs to be redrawn
	if(animFlag){
		glutPostRedisplay();
		animTime++;
	}
}
void fpsDisplay(){
	static int fpsCount=0;
	static int fpsLimit=1;
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
	axes[2]*=(w/(double)window.x);
	axes[3]*=(h/(double)window.y);

	window=make_int2(w,h);
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
void keyPressed(unsigned char key, int x, int y){
	switch(key){
	case 27:
		exit(0);
		break;
	case 'z':
		zoom(x, y, 0.920);
		break;
	case 'x':
		zoom(x, y, 1.087);
		break;
	case 'q':
		setROI(x, y, 0.5);
		break;
	case 'e':
		setROI(x, y, 2.0);
		break;
	case 'w':
		poi.y=min(poi.y+(roi.y+31)/32, height-roi.y);
		break;
	case 'a':
		poi.x=max(poi.x-(roi.x+31)/32, 0);
		break;
	case 's':
		poi.y=max(poi.y-(roi.y+31)/32, 0);
		break;
	case 'd':
		poi.x=min(poi.x+(roi.x+31)/32, width-roi.x);
		break;
	case 32: // toggle animation
		animFlag=!animFlag;
		break;
	}

	// indicate the display must be redrawn
	glutPostRedisplay();
}
void keyReleased(unsigned char key, int x, int y){

}
void mouseButton(int button, int state, int x, int y){
	mousePos.x=x;
	mousePos.y=y;
}
void mouseMotion(int x, int y){
	axes[0]-=(x-mousePos.x)*roi.x*axes[2]/(width*window.x);
	axes[1]-=(y-mousePos.y)*roi.y*axes[3]/(height*window.y);
	mousePos.x=x;
	mousePos.y=y;
}
void timerEvent(int value){
	glutPostRedisplay();
	glutTimerFunc(10, timerEvent, 0);
}
void idle(){
	glutPostRedisplay();
}

void createPBO(GLuint* pbo, int size){
	if(pbo){
		// set up vertex data parameter
		int size_tex_data = size*sizeof(uchar4);
		// Generate a buffer ID called a PBO (Pixel Buffer Object)
		glGenBuffers(1, pbo);
		// Make this the current UNPACK buffer (OpenGL is state-based)
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
		// Allocate data for the buffer. 4-channel 8-bit image
		glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
		cudaGLRegisterBufferObject(*pbo);
	}
}
void deletePBO(GLuint* pbo){
	if(pbo){
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);
		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);
		pbo=NULL;
		delete pbo;
	}
}
void createTexture(GLuint* textureID, int width, int height){
	// Enable Texturing
	glEnable(GL_TEXTURE_RECTANGLE_ARB);

	// Generate a texture identifier
	glGenTextures(1, textureID);

	// Make this the current texture (remember that GL is state-based)
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, *textureID);

	// Allocate the texture memory. The last parameter is NULL since we only
	// want to allocate memory, not initialize it
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA8, width, height, 0,
		GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	// Must set the filter mode, GL_LINEAR enables interpolation when scaling
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
	// GL_TEXTURE_2D for improved performance if linear interpolation is
	// not desired. Replace GL_LINEAR with GL_NEAREST in the
	// glTexParameteri() call
}
void deleteTexture(GLuint* tex){
	glDeleteTextures(1, tex);
	tex=NULL;
	delete tex;
}

void cleanupCuda(){
	if(pbo){
		deletePBO(&pbo);
	}
	if(textureID){
		deleteTexture(&textureID);
	}
}
bool initGL(int* argc, char **argv){
	// create a window and GL context (also register callbacks)
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window.x, window.y);
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

	// check for necessary OpenGL extensions
	glewInit();
	if(!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);
	return true;
}
void initCuda(int argc, char** argv){
	// First initialize OpenGL context, so we can properly set the GL
	// for CUDA.  NVIDIA notes this is necessary in order to achieve
	// optimal performance with OpenGL/CUDA interop.  use command-line
	// specified CUDA device, otherwise use device with highest Gflops/s
	cudaGLSetGLDevice(findCudaDevice(argc, (const char **)argv));
	createPBO(&pbo, width*height);
	createTexture(&textureID, width, height);

	// Clean up on program exit
	atexit(cleanupCuda);
	runCuda();
}

void animation(int argc, char** argv, int w, int h, void (*fun)(int, int, uchar4*, double, double, double, double)){
	width=w;
	height=h;
	render=fun;

	axes[0]=axes[1]=-2;
	axes[2]=axes[3]=4;
	if(w>h){
		axes[2]*=w/(double)h;
	}else if(h>w){
		axes[3]*=h/(double)w;
	}

	poi=make_int2(0,0);
	roi=make_int2(w,h);
	window=make_int2(clamp(w,720,1920), clamp(h,720,1080));

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

#endif /* ANIMATION_H_ */
