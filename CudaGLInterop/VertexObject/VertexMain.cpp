// simpleGLmain.cpp (Rob Farber)

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_timer.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>
#include "arcball.h"


// The user must create the following routines:
// CUDA methods
extern void initCuda(int argc, char** argv);
extern void runCuda();
extern void renderCuda(int);

// callbacks
extern void display();
extern void reshape(int, int);
extern void keyboard(unsigned char, int, int);
extern void mouse(int, int, int, int);
extern void motion(int, int);
extern void idle();

// GLUT specific variables
int2 window = make_int2(720, 720);
// Timer for FPS calculations
StopWatchInterface *timer = NULL; 
int fpsCount = 0;
int fpsLimit = 100;

// Simple method to display the Frames Per Second in the window title
void computeFPS(){
	fpsCount++;
	if(fpsCount==fpsLimit){
		char fps[256];
		float time=sdkGetAverageTimerValue(&timer);
		float ifps=1000.f/sdkGetAverageTimerValue(&timer);
		sprintf_s(fps, "Cuda GL Interop Wrapper: %3.1f fps ", ifps);
		glutSetWindowTitle(fps);
		fpsCount=0;
		sdkResetTimer(&timer);
	}
}

void fpsDisplay(){
	sdkStartTimer(&timer);
	display();
	sdkStopTimer(&timer);
	computeFPS();
}

bool initGL(int argc, char **argv){
	// Steps 1-2: create a window and GL context (also register callbacks)
	arcball_reset();
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(window.x, window.y);
	glutCreateWindow("Cuda GL Interop Demo (adapted from NVIDIA's simpleGL");
	glutDisplayFunc(fpsDisplay);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutMotionFunc(motion);
	glutIdleFunc(idle);

	// check for necessary OpenGL extensions
	glewInit();
	if(!glewIsSupported("GL_VERSION_2_0 ")){
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	return true;
}

// Main program
int main(int argc, char** argv){
	sdkCreateTimer(&timer);

	if(!initGL(argc, argv)){
		return EXIT_FAILURE;
	}

	initCuda(argc, argv);
	//SDK_CHECK_ERROR_GL();
	
	// register callbacks
	glutDisplayFunc(fpsDisplay);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	// start rendering mainloop
	glutMainLoop();

	// clean up
	sdkDeleteTimer(&timer);
	cudaThreadExit();
	exit(EXIT_SUCCESS);
}