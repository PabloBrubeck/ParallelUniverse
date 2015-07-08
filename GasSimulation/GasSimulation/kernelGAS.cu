#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <helper_cuda.h>
#include <cutil_math.h>
#include "geometry.h"
#include "linalg.h"

#define MAXTHREADS 512
#define L 4*0.015625f               // cell lenght = 2*particle radius
#define L2 L*L                    // cell lenght squared
#define cells 33                 // number of cells in one dimension
#define cells3 cells*cells*cells  // number of cells in three dimensions

struct Particle{
	float3 pos;
	float3 vel;
	float3 acc;
};
__device__
int cellIndex(float3 pos){
	int i=(int)((pos.x+1.f)/L);
	int j=(int)((pos.y+1.f)/L);
	int k=(int)((pos.z+1.f)/L);
	return (k*cells+j)*cells+i;
}
__device__
void boundaries(Particle &p, float3 pos, float3 vel){
	// 2x2x2 box fixed boundaries
	if(abs(pos.x)>1.f){
		vel.x=-vel.x;
	}
	if(abs(pos.y)>1.f){
		vel.y=-vel.y;
	}
	if(abs(pos.z)>1.f){
		vel.z=-vel.z;
	}

	pos=clamp(pos, -1.f, 1.f);
	p.pos=pos;
	p.vel=vel;
}
__device__ 
void collision(Particle *d_gas, int a, int b){
	float3 s=d_gas[a].pos-d_gas[b].pos;
	float3 u=d_gas[a].vel-d_gas[b].vel;
	float uu=dot(u, u);
	float su=dot(s, u);
	float ss=dot(s, s);
	float t0=-(su+sqrtf(su*su-uu*(ss-L2)))/uu;
	float3 r=s+u*t0;            // |r|=L always
	float3 du=r*(dot(u, r)/L2);
	float3 ds=du*t0;
	boundaries(d_gas[a], d_gas[a].pos+ds, d_gas[a].vel+du);
	boundaries(d_gas[b], d_gas[b].pos-ds, d_gas[b].vel-du);
}

__global__
void indices(uint4* d_index, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;
		
		int ii=(i+1)%mesh.x;
		int jj=(j+1)%mesh.y;
		int a=(k*mesh.y+j)*mesh.x+i; 
		int b=(k*mesh.y+j)*mesh.x+ii; 
		int c=(k*mesh.y+jj)*mesh.x+ii; 
		int d=(k*mesh.y+jj)*mesh.x+i;
		d_index[gid]=make_uint4(a, b, c, d);
	}
}
__global__
void initialState(Particle* d_gas, dim3 mesh){
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k=blockIdx.z*blockDim.z+threadIdx.z;
	if(i<mesh.x && j<mesh.y && k<mesh.z){
		int gid=(k*mesh.y+j)*mesh.x+i;

		float x=(float(2*i))/mesh.x-1.f;
		float y=(float(2*j))/mesh.y-1.f;
		float z=(float(2*k))/mesh.z-1.f;
		
		d_gas[gid].pos={x, y, z};
		d_gas[gid].vel={-x, -y, -z};
		d_gas[gid].acc={0.f, 0.f, 0.f};
	}
}
__global__ 
void updateGrid(Particle* d_gas, uint* d_gridCounters, uint* d_gridCells, int n){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	if(gid<n){
		// Update grid
		int cid=cellIndex(d_gas[gid].pos);
		int s=atomicInc(&(d_gridCounters[cid]), 1u);
		if(s<4u){
			d_gridCells[4*cid+s]=gid;
		}
	}
}
__global__
void neighbors(Particle* d_gas, uint* d_gridCounters, uint* d_gridCells){
	int hx=blockIdx.x*blockDim.x+threadIdx.x;
	int hy=blockIdx.y*blockDim.y+threadIdx.y;
	int hz=blockIdx.z*blockDim.x+threadIdx.z;
	if(hx<cells && hy<cells && hz<cells){
		int hid=(hz*cells+hy)*cells+hx;
		int hcount=d_gridCounters[hid];
		
		if(hcount==0){
			return;
		}
		int ncount=0;
		int neighbors[128];

		int nx, ny, nz;
		for(int i=-1; i<=1; i++){
			nx=hx+i;
			for(int j=-1; j<=1; j++){
				ny=hy+j;
				for(int k=-1; k<=1; k++){					
					nz=hz+k;
					if(	(nx>=0 && nx<cells) && 
						(ny>=0 && ny<cells) && 
						(nz>=0 && nz<cells)){
						int nid=(nz*cells+ny)*cells+nx;
						int acount=d_gridCounters[nid];
						for(int m=0; m<acount; m++){
							neighbors[ncount++]=d_gridCells[4*nid+m];
						}
					}
				}
			}
		}
		
		int home, away;
		for(int h=0; h<hcount; h++){
			home=d_gridCells[4*hid+h];
			float3 posh=d_gas[home].pos;
			for(int a=0; a<ncount; a++){
				away=neighbors[a];
				if(home!=away){
					// Check if particles are close enough
					float3 posn=d_gas[away].pos;
					float3 r=posh-posn;
					float r2=dot(r,r);
					if(r2<L2){
						// Check if the barycenter belongs to home
						float3 b=(posh+posn)/2.f;
						if(cellIndex(b)==hid){
							collision(d_gas, home, away);
						}
					}
				}
			}
		}			
	}
}
__global__
void integrate(Particle* d_gas, float step, int n){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	if(gid<n){
		float3 vel=d_gas[gid].vel+d_gas[gid].acc*step;
		float3 pos=d_gas[gid].pos+vel*step;
		boundaries(d_gas[gid], pos, vel);
	}
}
__global__
void updatePoints(float4 *d_pos, float4 *d_norm, uchar4 *d_color, Particle *d_gas, int n){
	int gid=blockIdx.x*blockDim.x+threadIdx.x;
	if(gid<n){
		float3 pos=d_gas[gid].pos;
		float3 vel=d_gas[gid].vel;
		float3 N=normalize(vel);

		d_pos[gid]=make_float4(pos.x, pos.y, pos.z, 1.f);
		d_norm[gid]=make_float4(N.x, N.y, N.z, 0.f);

		float x=N.x,  y=N.y,  z=N.z;
		float x2=x*x, y2=y*y, z2=z*z;
		
		float r=(x<0?0:x2)+(y<0?y2:0)+(z<0?z2:0);
		float g=(x<0?x2:0)+(y<0?0:y2)+(z<0?z2:0);
		float b=(x<0?x2:0)+(y<0?y2:0)+(z<0?0:z2);
		
		d_color[gid].x=(unsigned char)(255.f*r)&0xff;
		d_color[gid].y=(unsigned char)(255.f*g)&0xff;
		d_color[gid].z=(unsigned char)(255.f*b)&0xff;
		d_color[gid].w=255u;
	}
}

inline uint ceil(uint num, uint den){
	return (num+den-1u)/den;
}

void launch_kernel(float4 *d_pos, float4 *d_norm, uchar4 *d_color, uint4 *d_index, dim3 mesh, float time){
	static const int n=mesh.x*mesh.y*mesh.z;
	static const int bpg=ceil(cells, 8);
	static const dim3 block1D(MAXTHREADS);
	static const dim3 grid1D(ceil(n, MAXTHREADS));
	static const dim3 block3D(8, 8, 8);
	static const dim3 grid3D(bpg, bpg, bpg);
	static Particle *d_gas=NULL;
	static uint *d_gridCounters=NULL, *d_gridCells=NULL;

	if(d_gas==NULL){
		checkCudaErrors(cudaMalloc((void**)&d_gas, n*sizeof(Particle)));
		checkCudaErrors(cudaMalloc((void**)&d_gridCounters, cells3*sizeof(uint)));
		checkCudaErrors(cudaMalloc((void**)&d_gridCells, 4*cells3*sizeof(uint)));
		dim3 tgrid(ceil(mesh.x, block3D.x), ceil(mesh.y, block3D.y), ceil(mesh.z, block3D.z));
		initialState<<<tgrid, block3D>>>(d_gas, mesh);
		indices<<<tgrid, block3D>>>(d_index, mesh);
	}

	checkCudaErrors(cudaMemset(d_gridCounters, 0u, cells3*sizeof(uint)));
	checkCudaErrors(cudaMemset(d_gridCells, 0u, 4*cells3*sizeof(uint)));
	
	static float step=0.001f;
	integrate   <<<grid1D, block1D>>>(d_gas, step, n);
	updateGrid  <<<grid1D, block1D>>>(d_gas, d_gridCounters, d_gridCells, n);
	neighbors   <<<grid3D, block3D>>>(d_gas, d_gridCounters, d_gridCells);
	updatePoints<<<grid1D, block1D>>>(d_pos, d_norm, d_color, d_gas, n);

	cudaThreadSynchronize();
	checkCudaErrors(cudaGetLastError());
}