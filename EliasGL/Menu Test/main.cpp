#include <windows.h>
#include <gl/gl.h>
#include <GL/glut.h>
#include <math.h>
#define PI 3.14159265
#define DEG_TO_RAD 0.017453 /* degrees to radians */
GLfloat theta=0.0;


void mykey(unsigned char key, int x, int y)
{
    if(key == 'q' || key == 'Q' || key == '\27')
        exit(0);
}

void displayTriangle()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glBegin(GL_TRIANGLES);
        glColor3f(1.0, 1.0, 1.0);
        glVertex2f(-0.5, 0.0);
        glVertex2f(0.5, 0.0);
        glVertex2f(0.0, 0.5);
    glEnd();
    glFlush();
}

void display()
{
    //Clear window
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.0, 0.0, 0.0, 0.0);

    //Draws
    glBegin(GL_TRIANGLE_FAN);
        glColor3f(0.0, 0.0, 0.0);
        glVertex2f(0.0, 0.0);
        for(int i=0; i<90; i++)
        {
            glColor3f(0.0, 1.0, 0.0);
            glVertex2f(0.5*cos(i*PI/180.0), 0.5*sin(i*PI/180));
        }
        for(int i=90; i<180; i++)
        {
            glColor3f(1.0, 0.0, 0.0);
            glVertex2f(0.5*cos(i*PI/180.0), 0.5*sin(i*PI/180));
        }
        for(int i=180; i<270; i++)
        {
            glColor3f(1.0, 1.0, 0.0);
            glVertex2f(0.5*cos(i*PI/180.0), 0.5*sin(i*PI/180));
        }
        for(int i=270; i<361; i++)
        {
            glColor3f(0.0, 0.0, 1.0);
            glVertex2f(0.5*cos(i*PI/180.0), 0.5*sin(i*PI/180));
        }
    glEnd();

    //Flush GL buffers.
    glFlush();

}

void mymenu(int value)
{
    if(value==0)
        displayTriangle();
    if(value==1)
        display();
    if(value==2)
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glFlush();
    }
    if(value==3)
        exit(0);
}

void init()
{
    //Set clear color.
    glClearColor(0.0, 0.0, 0.0, 0.0);
    //Set fill color
    glColor3f(1.0, 1.0, 1.0);
    glLineWidth(5.0);
    glEnable(GL_LINE_STIPPLE);
    glLineStipple(3, 0xcccc); //

    /* Set up standard orthogonal view with clipping
        box as cube of side 2 centered at origin
        this is default view and these statements could be removed
    */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0,0);
    glutCreateWindow("simple");
    glutDisplayFunc(displayTriangle);
    glutKeyboardFunc(mykey);
    int id = glutCreateMenu(mymenu);
    glutAddMenuEntry("Triangle", 0);
    glutAddMenuEntry("Circle", 1);
    glutAddMenuEntry("Clear screen", 2);
    glutAddMenuEntry("Exit", 3);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
    init();
    glutMainLoop();
}
