#define _DEBUG

#include <ctype.h>
#include <PhysX/PxPhysicsAPI.h>
#include <PhysX/foundation/PxTransform.h>

using namespace physx;

class Camera
{
public:
	Camera(const PxVec3 &eye, const PxVec3& dir);

	void			handleMouse(int button, int state, int x, int y);
	bool			handleKey(unsigned char key, int x, int y, float speed = 1.0f);
	void			handleMotion(int x, int y);
	void			handleAnalogMove(float x, float y);

	PxVec3			getEye()		const;
	PxVec3			getDir()		const;
	PxTransform		getTransform()	const;
private:
	PxVec3			mEye;
	PxVec3			mDir;
	int				mMouseX;
	int				mMouseY;
};

Camera::Camera(const PxVec3 &eye, const PxVec3& dir)
{
	mEye = eye;
	mDir = dir.getNormalized();
	mMouseX = 0;
	mMouseY = 0;
}

void Camera::handleMouse(int button, int state, int x, int y)
{
	PX_UNUSED(state);
	PX_UNUSED(button);
	mMouseX = x;
	mMouseY = y;
}

bool Camera::handleKey(unsigned char key, int x, int y, float speed)
{
	PX_UNUSED(x);
	PX_UNUSED(y);

	PxVec3 viewY = mDir.cross(PxVec3(0,1,0)).getNormalized();
	switch(toupper(key))
	{
	case 'W':	mEye += mDir*2.0f*speed;		break;
	case 'S':	mEye -= mDir*2.0f*speed;		break;
	case 'A':	mEye -= viewY*2.0f*speed;		break;
	case 'D':	mEye += viewY*2.0f*speed;		break;
	default:							return false;
	}
	return true;
}

void Camera::handleAnalogMove(float x, float y)
{
	PxVec3 viewY = mDir.cross(PxVec3(0,1,0)).getNormalized();
	mEye += mDir*y;
	mEye += viewY*x;
}

void Camera::handleMotion(int x, int y)
{
	int dx = mMouseX - x;
	int dy = mMouseY - y;

	PxVec3 viewY = mDir.cross(PxVec3(0,1,0)).getNormalized();

	PxQuat qx(PxPi * dx / 180.0f, PxVec3(0,1,0));
	mDir = qx.rotate(mDir);
	PxQuat qy(PxPi * dy / 180.0f, viewY);
	mDir = qy.rotate(mDir);

	mDir.normalize();

	mMouseX = x;
	mMouseY = y;
}

PxTransform Camera::getTransform() const
{
	PxVec3 viewY = mDir.cross(PxVec3(0,1,0));

	if(viewY.normalize()<1e-6f) 
		return PxTransform(mEye);

	PxMat33 m(mDir.cross(viewY), viewY, -mDir);
	return PxTransform(mEye, PxQuat(m));
}

PxVec3 Camera::getEye() const
{ 
	return mEye; 
}

PxVec3 Camera::getDir() const
{ 
	return mDir; 
}