#include <cutil_math.h>

__device__
const uchar4 jetmap[5]={
	{0x00, 0x00, 0x7F, 0xFF}, //blue
	{0x00, 0x7F, 0xFF, 0xFF}, //cyan
	{0x7F, 0xFF, 0x7F, 0xFF}, //green
	{0xFF, 0x7F, 0x00, 0xFF}, //orange
	{0x7F, 0x00, 0x00, 0xFF}  //red
};

__device__
const uchar4 hotmap[5]={
		{0x00, 0x00, 0x00, 0xFF}, //black
		{0xFF, 0x00, 0x00, 0xFF}, //red
		{0xFF, 0x7F, 0x00, 0xFF}, //orange
		{0xFF, 0xFF, 0x00, 0xFF}, //yellow
		{0xFF, 0xFF, 0xFF, 0xFF}  //white
};

__device__ uchar4 lerp(uchar4 a, uchar4 b, float t){
	return make_uchar4(lerp(a.x, b.x, t),
			lerp(a.y, b.y, t),
			lerp(a.z, b.z, t),
			lerp(a.w, b.w, t));
}

__global__ void gray(uchar4 *d_cmap, int n){
	int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<n){
		unsigned char gs=(256*gid)/n;
		d_cmap[gid]=make_uchar4(gs, gs, gs, 0xFF);
	}
}
__global__ void jet(uchar4 *d_cmap, int n){
	int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<n){
		int i=(4*gid)/n;
		d_cmap[gid]=lerp(jetmap[i], jetmap[i+1], (4.f*gid)/n-i);
	}
}
__global__ void hot(uchar4 *d_cmap, int n){
	int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<n){
		int i=(4*gid)/n;
		d_cmap[gid]=lerp(hotmap[i], hotmap[i+1], (4.f*gid)/n-i);
	}
}


__global__ void quadgray(uchar4 *d_cmap, int n){
	int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<n){
		unsigned char gs=256.f*sqrtf(float(gid)/n);
		d_cmap[gid]=make_uchar4(gs, gs, gs, 0xFF);
	}
}
__global__ void quadjet(uchar4 *d_cmap, int n){
	int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<n){
		int i=(4*gid)/n;
		d_cmap[gid]=lerp(jetmap[i], jetmap[i+1], sqrtf((4.f*gid)/n-i));
	}
}
__global__ void quadhot(uchar4 *d_cmap, int n){
	int gid=blockDim.x*blockIdx.x+threadIdx.x;
	if(gid<n){
		int i=(4*gid)/n;
		d_cmap[gid]=lerp(hotmap[i], hotmap[i+1], sqrtf((4.f*gid)/n-i));
	}
}
