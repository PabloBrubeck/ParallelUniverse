// simpleGLmain.cpp (Rob Farber)

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_timer.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>


// The user must create the following routines:
// CUDA methods
extern void initCuda(int argc, char** argv);
extern void runCuda();
extern void renderCuda(int);

// callbacks
extern void display();
extern void keyboard(unsigned char key, int x, int y);
extern void mouse(int button, int state, int x, int y);
extern void motion(int x, int y);

// GLUT specific variables
unsigned int window_width  = 512;
unsigned int window_height = 512;

// Timer for FPS calculations
StopWatchInterface *timer = NULL; 
int fpsCount = 0;
int fpsLimit = 100;

// Simple method to display the Frames Per Second in the window title
void computeFPS(){
	fpsCount++;
	if(fpsCount==fpsLimit){
		char fps[256];
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
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop Demo (adapted from NVIDIA's simpleGL");
	glutDisplayFunc(fpsDisplay);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);

	// check for necessary OpenGL extensions
	glewInit();
	if(!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// Step 3: Setup our viewport and viewing modes
	glViewport(0, 0, window_width, window_height);

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);
	
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