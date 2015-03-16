#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>

__device__
float param(float min, float max, char s, int i, int n){
	float range=max-min;
	switch(s){
	default:
	case 0: // (min, max)
		return min+range*((i+0.5f)/n);
	case 1: // [min, max)
		return min+range*((i+0.0f)/n);
	case 2: // (min, max]
		return min+range*((i+1.0f)/n);
	case 3: // [min, max]
		return min+range*(i/(n-1.f));
	}
}

__device__
void cylindrical(float4 &p, float r, float theta, float z){
	p.x=r*cosf(theta);
	p.y=r*sinf(theta);
	p.z=z;
}
__device__
void spherical(float4 &p, float rho, float theta, float phi){
	float r=rho*sinf(phi);
	p.x=r*cosf(theta);
	p.y=r*sinf(theta);
	p.z=rho*cosf(phi);
}
__device__
void torus(float4 &p, float u, float v, float a, float c){
	float r=c+a*cosf(v);
	p.x=r*cosf(u);
	p.y=r*sinf(u);
	p.z=a*sinf(v);
}
__device__
void mobius(float4 &p, float s, float t, float r){
	r+=s*cosf(t/2.f);
    p.x=r*cosf(t);
    p.y=r*sinf(t);
    p.z=s*sinf(t/2.f);
}
__device__
void figureEight(float4 &p, float u, float v, float r, float c){
	float a=sinf(v);
    float b=sinf(2*v);
    float cos=cosf(-u);
    float sin=sinf(-u);
    float t=r*(c+a*cos-b*sin);
    p.x=t*cosf(-2*u);
	p.y=t*sinf(-2*u);
	p.z=r*(a*sin+b*cos);
}