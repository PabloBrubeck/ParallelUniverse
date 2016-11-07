/*
 * kernel.h
 *
 *  Created on: May 31, 2016
 *      Author: pbrubeck
 */

#ifndef KERNEL_H_
#define KERNEL_H_

#define MAXTHREADS 512

template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

inline int nextPow2(int v){
	v--;
	v|=v>>1;
	v|=v>>2;
	v|=v>>4;
	v|=v>>8;
	v|=v>>16;
	v++;
	return v;
}
inline int ceil(int num, int den){
	return (num+den-1)/den;
}
inline dim3 grid(int i){
	return dim3(ceil(i,MAXTHREADS), 1, 1);
}
inline dim3 grid(int i, int j){
	return dim3(ceil(i,MAXTHREADS), j, 1);
}
inline dim3 grid(int i, int j, int k){
	return dim3(ceil(i,MAXTHREADS), j, k);
}

inline void gridblock(dim3 &grid, dim3 &block, dim3 mesh){
	block.x=min(mesh.x, MAXTHREADS);
	block.y=min(mesh.y, MAXTHREADS/(block.x));
	block.z=min(mesh.z, MAXTHREADS/(block.x*block.y));
	grid.x=ceil(mesh.x, block.x);
	grid.y=ceil(mesh.y, block.y);
	grid.z=ceil(mesh.z, block.z);
}

#endif /* KERNEL_H_ */
