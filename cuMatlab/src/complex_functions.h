/*
 * complex_functions.h
 *
 *  Created on: Aug 2, 2016
 *      Author: pbrubeck
 */

#ifndef COMPLEX_FUNCTIONS_H_
#define COMPLEX_FUNCTIONS_H_

#include "cuComplex.h"

/*
 * Double Precision
 */

#define M_I make_cuDoubleComplex(0,1)

__host__ __device__ static __inline__
cuDoubleComplex operator-(cuDoubleComplex z){
    return make_cuDoubleComplex(-z.x, -z.y);
}

__host__ __device__ static __inline__
cuDoubleComplex operator+(cuDoubleComplex z, cuDoubleComplex w){
    return cuCadd(z,w);
}

__host__ __device__ static __inline__
cuDoubleComplex operator-(cuDoubleComplex z, cuDoubleComplex w){
    return cuCsub(z,w);
}

__host__ __device__ static __inline__
cuDoubleComplex operator*(cuDoubleComplex z, cuDoubleComplex w){
    return cuCmul(z,w);
}

__host__ __device__ static __inline__
cuDoubleComplex operator/(cuDoubleComplex z, cuDoubleComplex w){
    return cuCdiv(z,w);
}

__host__ __device__ static __inline__
cuDoubleComplex operator+(cuDoubleComplex z, double t){
    return make_cuDoubleComplex(z.x+t, z.y);
}

__host__ __device__ static __inline__
cuDoubleComplex operator-(cuDoubleComplex z, double t){
	return make_cuDoubleComplex(z.x-t, z.y);
}

__host__ __device__ static __inline__
cuDoubleComplex operator*(cuDoubleComplex z, double t){
	return make_cuDoubleComplex(z.x*t, z.y*t);
}

__host__ __device__ static __inline__
cuDoubleComplex operator/(cuDoubleComplex z, double t){
	return make_cuDoubleComplex(z.x/t, z.y/t);
}

__host__ __device__ static __inline__
cuDoubleComplex operator+(double t, cuDoubleComplex z){
    return make_cuDoubleComplex(t+z.x, z.y);
}

__host__ __device__ static __inline__
cuDoubleComplex operator-(double t, cuDoubleComplex z){
	return make_cuDoubleComplex(t-z.x, -z.y);
}

__host__ __device__ static __inline__
cuDoubleComplex operator*(double t, cuDoubleComplex z){
	return make_cuDoubleComplex(t*z.x, t*z.y);
}

__host__ __device__ static __inline__
cuDoubleComplex operator/(double t, cuDoubleComplex z){
	double z2=z.x*z.x+z.y*z.y;
	return make_cuDoubleComplex(t*z.x/z2, -t*z.y/z2);
}

__host__ __device__ static __inline__
cuDoubleComplex conj(cuDoubleComplex z){
    return cuConj(z);
}

__host__ __device__ static __inline__
double abs(cuDoubleComplex z){
    return cuCabs(z);
}

static __inline__ __host__ __device__
double angle(cuDoubleComplex z){
	return atan2(z.y, z.x);
}

static __inline__ __host__ __device__
cuDoubleComplex sqrt(cuDoubleComplex z){
	double r=sqrt(cuCabs(z));
	double t=0.5*atan2(z.y, z.x);
	double s, c; sincos(t, &s, &c);
	return make_cuDoubleComplex(r*c, r*s);
}

static __inline__ __host__ __device__
cuDoubleComplex exp(cuDoubleComplex z){
	double r=exp(z.x);
	double s, c; sincos(z.y, &s, &c);
	return make_cuDoubleComplex(r*c, r*s);
}

static __inline__ __host__ __device__
cuDoubleComplex log(cuDoubleComplex z){
	return make_cuDoubleComplex(0.5*log(z.x*z.x+z.y*z.y), atan2(z.y, z.x));
}

static __inline__ __host__ __device__
cuDoubleComplex pow(cuDoubleComplex z, cuDoubleComplex w){
	return exp(log(z)*w);
}

static __inline__ __host__ __device__
cuDoubleComplex pow(cuDoubleComplex z, double t){
	return exp(log(z)*t);
}

static __inline__ __host__ __device__
cuDoubleComplex pow(double t, cuDoubleComplex z){
	return exp(log(t)*z);
}

static __inline__ __host__ __device__
cuDoubleComplex sin(cuDoubleComplex z){
	double s, c; sincos(z.x, &s, &c);
	return make_cuDoubleComplex(s*cosh(z.y), c*sinh(z.y));
}

static __inline__ __host__ __device__
cuDoubleComplex cos(cuDoubleComplex z){
	double s, c; sincos(z.x, &s, &c);
	return make_cuDoubleComplex(c*cosh(z.y), -s*sinh(z.y));
}

static __inline__ __host__ __device__
cuDoubleComplex tan(cuDoubleComplex z){
	double s, c; sincos(2*z.x, &s, &c);
	c+=cosh(2*z.y);
	return make_cuDoubleComplex(s/c, sinh(2*z.y)/c);
}

static __inline__ __host__ __device__
cuDoubleComplex sinh(cuDoubleComplex z){
	double s, c; sincos(z.y, &s, &c);
	return make_cuDoubleComplex(c*sinh(z.x), s*cosh(z.x));
}

static __inline__ __host__ __device__
cuDoubleComplex cosh(cuDoubleComplex z){
	double s, c; sincos(z.y, &s, &c);
	return make_cuDoubleComplex(c*cosh(z.x), s*sinh(z.x));
}

static __inline__ __host__ __device__
cuDoubleComplex tanh(cuDoubleComplex z){
	double s, c; sincos(2*z.y, &s, &c);
	c+=cosh(2*z.x);
	return make_cuDoubleComplex(sinh(2*z.x)/c, s/c);
}

static __inline__ __host__ __device__
cuDoubleComplex asin(cuDoubleComplex z){
	z=make_cuDoubleComplex(-z.y, z.x);
	cuDoubleComplex w=log(z+sqrt(z*z+1));
	return make_cuDoubleComplex(w.y, -w.x);
}

static __inline__ __host__ __device__
cuDoubleComplex acos(cuDoubleComplex z){
	cuDoubleComplex w=log(z+sqrt(z*z-1));
	return make_cuDoubleComplex(w.y, -w.x);
}

static __inline__ __host__ __device__
cuDoubleComplex atan(cuDoubleComplex z){
	z=make_cuDoubleComplex(-z.y, z.x);
	cuDoubleComplex w=0.5*log((1+z)/(1-z));
	return make_cuDoubleComplex(w.y, -w.x);
}

static __inline__ __host__ __device__
cuDoubleComplex asinh(cuDoubleComplex z){
	return log(z+sqrt(z*z+1));
}

static __inline__ __host__ __device__
cuDoubleComplex acosh(cuDoubleComplex z){
	return log(z+sqrt(z*z-1));
}

static __inline__ __host__ __device__
cuDoubleComplex atanh(cuDoubleComplex z){
	return 0.5*log((1+z)/(1-z));
}


#endif /* COMPLEX_FUNCTIONS_H_ */
