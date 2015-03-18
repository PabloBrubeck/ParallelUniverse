#include <windows.h>
#include <gl/gl.h>
#include <GL/glut.h>
#include <math.h>
#define DEG_TO_RAD 0.017453
int singleb, doubleb; //Window ids
GLfloat theta = 0.0;

void displays()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POLYGON);
        glVertex2f(sin(DEG_TO_RAD*theta), cos(DEG_TO_RAD*theta));
        glVertex2f(-sin(DEG_TO_RAD*theta), cos(DEG_TO_RAD*theta));
        glVertex2f(-sin(DEG_TO_RAD*theta), -cos(DEG_TO_RAD*theta));
        glVertex2f(sin(DEG_TO_RAD*theta), -cos(DEG_TO_RAD*theta));
    glEnd();
    glFlush();
}

void displayd()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POLYGON);
        glVertex2f(sin(DEG_TO_RAD*theta), cos(DEG_TO_RAD*theta));
        glVertex2f(-sin(DEG_TO_RAD*theta), cos(DEG_TO_RAD*theta));
        glVertex2f(-sin(DEG_TO_RAD*theta), -cos(DEG_TO_RAD*theta));
        glVertex2f(sin(DEG_TO_RAD*theta), -cos(DEG_TO_RAD*theta));
    glEnd();
    glutSwapBuffers();
}

void spinDisplay()
{
    //Increment angle
    theta+=2.0;
    if(theta>360.0) theta=0;

    //Draw single buffer window
    glutSetWindow(singleb);
    glutPostWindowRedisplay(singleb);

    //Draw double buffer window
    glutSetWindow(doubleb);
    glutPostWindowRedisplay(doubleb);
}

void mouse(int btn, int state, int x, int y)
{
    if(btn == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
        glutIdleFunc(spinDisplay);
    if(btn == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN)
        glutIdleFunc(NULL);
}

void myReshape(int w, int h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-2.0, 2.0, -2.0, 2.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void mykey(unsigned char key, int x, int y)
{
    if(key == 'q' || key == 'Q' || key == '\27')
        exit(0);
}


void quit_menu(int id)
{
    if(id == 1) exit(0);
}

void myinit()
{

}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);

    //Create a single buffered window
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    singleb = glutCreateWindow("Single buffered");
    myinit();
    glutDisplayFunc(displays);
    glutReshapeFunc(myReshape);
    glutIdleFunc(spinDisplay);
    glutMouseFunc(mouse);
    glutKeyboardFunc(mykey);

    //Create double buffered window to right
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowPosition(310, 0);
    doubleb = glutCreateWindow("Double buffered");
    myinit();
    glutDisplayFunc(displayd);
    glutReshapeFunc(myReshape);
    glutIdleFunc(spinDisplay);
    glutMouseFunc(mouse);
    glutCreateMenu(quit_menu);
    glutAddMenuEntry("quit", 1);
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    //event loop
    glutMainLoop();

}
