#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cutil_math.h>

inline __host__ __device__ 
int sign(int x){
	return (x>0)-(x<0);
}
inline __host__ __device__ 
float4 cross(float4 &a, float4 &b){
	return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0.f);
}
inline __host__ __device__
void eye(float* A, int n){
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			A[i*n+j]=(i==j);
		}
	}
}
inline __host__ __device__ 
void mmult(float* C, float* A, float* B, int n){
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			float sum=0.f;
			for(int k=0; k<n; k++){
				sum+=A[i*n+k]*B[k*n+j];
			}
			C[i*n+j]=sum;
		}
	}
}
inline __host__ __device__ 
float4 vmult(const float* A, float4 &u){
	float x=u.x*A[0] +u.y*A[1] +u.z*A[2] +u.w*A[3];
	float y=u.x*A[4] +u.y*A[5] +u.z*A[6] +u.w*A[7];
	float z=u.x*A[8] +u.y*A[9] +u.z*A[10]+u.w*A[11];
	float w=u.x*A[12]+u.y*A[13]+u.z*A[14]+u.w*A[15];
	return make_float4(x, y, z, w);
}

//Rotations
void givens(float* Q, int i, int j, float theta){
	int k=3-i-j;
    float cos=cosf(theta);
    float sin=sign(i-j)*sinf(theta);
    Q[i*4+i]=cos;
    Q[j*4+j]=cos;
    Q[i*4+j]=sin;
    Q[j*4+i]=-sin;
    Q[k*4+k]=1.f;
}
void rotateXYZ(float* Q, float3 angle){
	float* Rx=new float[16];
	float* Ry=new float[16];
	float* Rz=new float[16];
	float* temp=new float[16];
	eye(Rx, 4);
	eye(Ry, 4);
	eye(Rz, 4);
	givens(Rx, 2, 1,  angle.x);
	givens(Ry, 2, 0, -angle.y);
	givens(Rz, 1, 0,  angle.z);
	mmult(temp, Ry, Rx, 4);
	mmult(Q, Rz, temp, 4);
	delete[] Rx, Ry, Rz, temp;
}
void rotateAxis(float* Q, float4 axis, float angle){
    const float c=cosf(angle);
	const float s=sinf(angle);
	const float x=axis.x;
    const float y=axis.y;
	const float z=axis.z;
	const float r=sqrtf(x*x+y*y+z*z);
	const float u[3]={x/r, y/r, z/r};
    int k=1;
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            Q[i*4+j]=(1-c)*u[i]*u[j]+(i==j? c: k*s*u[3-i-j]);
            k=(i==j || j==2)? k:-k;
        }
    }
}

__global__
void vmult(float4* d_out, const float* h_A, float4* d_in){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	d_out[gid]=vmult(h_A, d_in[gid]);
}

__global__
void tangent(float4* d_pos, float4* d_tan, int n){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	if(gid<n){
		d_tan[gid]=normalize(d_pos[gid+1]-d_pos[gid]);
	}
}

__global__
void frenet(float4* d_pos, float4* d_tan, float4* d_norm, float4* d_bin, int n){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	if(gid<n){
		d_norm[gid]=normalize(d_tan[gid+1]-d_tan[gid]);
		d_bin[gid]=cross(d_tan[gid], d_norm[gid]);
	}
}
