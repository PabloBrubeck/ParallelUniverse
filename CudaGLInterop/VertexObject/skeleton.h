#include <GL/glew.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <cutil_math.h>

#define PI 3.14159265f
#define deg2rad 0.017453f

float4 bones[25]={
	{0.0f, 0.0f, 0.0f, 1.f},
	{3.0f, 1.6f, 0.0f, 1.f},
	{2.6f, 4.7f, 0.0f, 1.f},
	{0.0f, 3.1f, 0.0f, 1.f},
	{0.0f, 2.4f, 0.0f, 1.f},
	{2.0f, 2.3f, 0.0f, 1.f},
	{0.7f, 5.4f, 0.0f, 1.f},
	{0.0f, 5.5f, 0.0f, 1.f},
	{0.0f, 2.5f, 0.0f, 1.f},
	{0.0f, 2.6f, 0.0f, 1.f},
	{1.2f, 2.4f, 0.0f, 1.f},
	{0.0f, 5.3f, 0.0f, 1.f},
	{0.0f, 6.5f, 0.0f, 1.f},
	{0.0f, 3.0f, 0.0f, 1.f},
	{0.0f, 2.8f, 0.0f, 1.f},
	{-0.4f, 2.3f, 0.0f, 1.f},
	{0.0f, 5.4f, 0.0f, 1.f},
	{0.0f, 5.5f, 0.0f, 1.f},
	{0.0f, 2.7f, 0.0f, 1.f},
	{0.0f, 3.1f, 0.0f, 1.f},
	{-1.1f, 2.4f, 0.0f, 1.f},
	{-0.4f, 5.1f, 0.0f, 1.f},
	{0.0f, 3.7f, 0.0f, 1.f},
	{0.0f, 2.2f, 0.0f, 1.f},
	{0.0f, 2.6f, 0.0f, 1.f},
};

int* fingers=new int[30]{
	0, 0,  1,  2,  3,  4,
	0, 5,  6,  7,  8,  9,
	0, 10, 11, 12, 13, 14,
	0, 15, 16, 17, 18, 19,
	0, 20, 21, 22, 23, 24,
};

int* path=new int[36]{
	1, 2, 3, 4, 3, 2,
	6, 7, 8, 9, 8, 7, 6,
	11, 12, 13, 14, 13, 12, 11,
	16, 17, 18, 19, 18, 17, 16,
	21, 22, 23, 24, 23, 22, 21, 20, 0
};

float3 angle[25]={
	{0.f, 0.f, 0.f}, //0 Thumb
	{10.f, 0.f, 0.f},
	{10.f, 0.f, 0.f},
	{10.f, 0.f, 0.f},
	{0.f, 0.f, 0.f},
	{0.f, 0.f, 0.f}, //5 Index
	{10.f, 0.f, 0.f},
	{10.f, 0.f, 0.f},
	{10.f, 0.f, 0.f},
	{0.f, 0.f, 0.f},
	{0.f, 0.f, 0.f}, //10 Middle
	{60.f, 0.f, 0.f},
	{60.f, 0.f, 0.f},
	{60.f, 0.f, 0.f},
	{0.f, 0.f, 0.f},
	{0.f, 0.f, 0.f}, //15 Ring
	{60.f, 0.f, 0.f},
	{60.f, 0.f, 0.f},
	{60.f, 0.f, 0.f},
	{0.f, 0.f, 0.f},
	{0.f, 0.f, 0.f}, //20 Pinky
	{60.f, 0.f, 0.f},
	{60.f, 0.f, 0.f},
	{60.f, 0.f, 0.f},
	{0.f, 0.f, 0.f},
};

int sign(int x){
	return (x>0)-(x<0);
}
void eye(float* A, int n){
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			A[i*n+j]=(i==j);
		}
	}
}
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
inline float4 vmult(float* A, float4 u){
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
	givens(Rx, 2, 1,  deg2rad*angle.x);
	givens(Ry, 2, 0, -deg2rad*angle.y);
	givens(Rz, 1, 0,  deg2rad*angle.z);
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

void skeleton(float4* skeleton, float3* angles){
	float* R=new float[16];
	float* Q=new float[16];
	for(int i=0; i<5; i++){
		eye(R, 4);
		for(int j=0; j<5; j++){
			int k=5*i+j;
			skeleton[k]=vmult(R, bones[k]);
			if(j>0){
				skeleton[k]+=skeleton[k-1];
			}
			skeleton[k].w=1.f;
			float* temp=new float[16];
			rotateXYZ(Q, angles[k]);
			mmult(temp, Q, R, 4);
			R=temp;
		}
	}
}
void volume(float4* hand, float4* skeleton, int n){
	float4* N=new float4[36];
	float4* B=new float4[36];
	float4 ez={0.f, 0.f, 1.f, 1.f};
	float4 zero={0.f, 0.f, 0.f, 1.f};
	for(int j=0; j<36; j++){
		int b=path[j];
		if(b%5==4){
			N[j]=zero;
			B[j]=zero;
		}else{
			int a=path[(j+35)%36];
			int c=path[(j+1)%36];
			float4 OA=skeleton[a];
			float4 OB=skeleton[b];
			float4 OC=skeleton[c];
			float4 AB=OB-OA;
			float4 BC=OC-OB;
			float4 AC=OC-OA;
			float4 n=cross(BC,AB);
			N[j]=normalize(dot(n,n)==0? cross(AC, ez): n);
			B[j]=normalize(cross(n, AC));
		}
	}
	float r=0.5f;
	for(int i=0; i<n; i++){
		float phi=(PI*i)/n;
		float x=r*cosf(phi);
		float y=r*sinf(phi);
		for(int j=0; j<36; j++){
			int k=i*36+j;
			int b=path[j];
			hand[k]=skeleton[b];
			if(b%5!=4){
				hand[k]+=x*N[j]+y*B[j];
			}
		}
	}
}

void moveFingerX(int finger, float delta){
	int index=5*finger+1;
	for(int i=0; i<3; i++){
		angle[index+i].x=clamp(angle[index+i].x+delta, 1.f, 90.f);
	}
}
void moveFingerZ(int finger, float delta){
	int index=5*finger+1;
	angle[index].z=clamp(angle[index].z+delta, -30.f, 30.f);
}