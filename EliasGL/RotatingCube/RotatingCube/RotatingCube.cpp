/* Rota con los botones del mouse*/

#include "stdafx.h"
#include <windows.h>
#include <glut.h>
#include <gl.h>

GLfloat colors[][3] = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 },
{ 1.0, 1.1, 0.0 }, { 0.0, 1.0, 1.0 }, { 1.0, 0.0, 1.0 } };
GLfloat vertices[][3] = { { -1.0, -1.0, 1.0 }, { -1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 },
{ 1.0, -1.0, 1.0 }, { -1.0, -1.0, -1.0 }, { -1.0, 1.0, -1.0 }, { 1.0, 1.0, -1.0 }, { 1.0, -1.0, -1.0 } };
int axis;
float theta[3];

void reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-4.0, 4.0, -4.0, 4.0, -4.0, 4.0);
}

void init()
{
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glColor3f(0.0, 0.0, 0.0);
}

void polygon(int a, int b, int c, int d)
{
	glBegin(GL_POLYGON);
	glVertex3fv(vertices[a]);
	glVertex3fv(vertices[b]);
	glVertex3fv(vertices[c]);
	glVertex3fv(vertices[d]);
	glEnd();
}

void colorcube()
{
	glColor3fv(colors[0]);
	polygon(0, 3, 2, 1);
	glColor3fv(colors[1]);
	polygon(2, 3, 7, 6);
	glColor3fv(colors[2]);
	polygon(3, 0, 4, 7);
	glColor3fv(colors[3]);
	polygon(1, 2, 6, 5);
	glColor3fv(colors[4]);
	polygon(4, 5, 6, 7);
	glColor3fv(colors[5]);
	polygon(5, 4, 0, 1);

}

void mouse(int btn, int state, int x, int y)
{
	if (btn == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
		axis = 0;
	if (btn == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN)
		axis = 1;
	if (btn == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
		axis = 2;
}

void spinCube()
{
	theta[axis] += 2.0;
	if (theta[axis] > 360.0) theta[axis] -= 360.0;
	glutPostRedisplay();
	Sleep(10);
}

void display(){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	gluLookAt(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glRotatef(theta[0], 1.0, 0.0, 0.0);
	glRotatef(theta[1], 0.0, 1.0, 0.0);
	glRotatef(theta[2], 0.0, 0.0, 1.0);
	colorcube();
	glutSwapBuffers();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Rotating Cube");
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutIdleFunc(spinCube);
	init();
	glutMainLoop();
}
