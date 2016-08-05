/*
 * examples.h
 *
 *  Created on: Aug 1, 2016
 *      Author: pbrubeck
 */

#ifndef EXAMPLES_H_
#define EXAMPLES_H_

void mapExample(int n){
	// Maps a function f(x) on the domain [0, pi]
	double *x;
	cudaMallocManaged((void**)&x, n*sizeof(double));
	linspace(0.0, pi, n, x);

	int m=1;
	int l=4;

	double *a;
	cudaMallocManaged((void**)&a, (l+1)*sizeof(double));
	cudaDeviceSynchronize();
	a[l]=1.0;

	auto Pml = [m, l, a] __device__ (double t){
		return LegendreP(l+1,a,m,cos(t));
	};
	cudaMap(Pml, n, x, x);
	cudaDeviceSynchronize();
	disp(n, x);
}

template<typename F>
void quadratureExample(F f, int n){
	// Computes the integral of f(x) on [-1,1]
	double *x=new double[n];
	double *w=new double[n];
	gauleg(n,x,w,-1,1);

	mapHost(f,n,x,x);
	int inc=1;
	double J=ddot(&n,w,&inc,x,&inc);
	printf("%.15f\n",J);
}

template<typename F>
void poisson(F f, double ua, double ub, int n){
	// Solves the Dirichlet problem u_xx = f(x), u(a)=ua, u(b)=ub
	double *x, *D, *D2;
	cudaMalloc((void**)&x, n*sizeof(double));
	cudaMalloc((void**)&D, n*n*sizeof(double));
	cudaMalloc((void**)&D2, n*n*sizeof(double));
	chebD(n, D, x);

	// compute second derivative operator D2=D*D
	cublasHandle_t cublasH;
	cublasCreate(&cublasH);
	double alpha=1, beta=0;
	cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, D, n, D, n, &beta, D2, n);

	// right hand side, unified memory!
	double *u;
	cudaMallocManaged((void**)&u, n*sizeof(double));
	cudaMap(f, n, x, u);

	// apply boundary conditions
	alpha=-ua;
	cublasDaxpy(cublasH, n, &alpha, D2+n*(n-1), 1, u, 1);
	alpha=-ub;
	cublasDaxpy(cublasH, n, &alpha, D2        , 1, u, 1);
	cudaDeviceSynchronize();
	u[0]=ub; u[n-1]=ua;

	// solve Poisson D2*u=f(x)
	cusolverDnHandle_t cusolverH;
	cusolverDnCreate(&cusolverH);
	linsolve(cusolverH, n-2, D2+n+1, n, u+1, n, 1);

	// display u
	cudaDeviceSynchronize();
	disp(n, u);

	// free device memory
	cudaFree(x);
	cudaFree(D);
	cudaFree(D2);
	cudaFree(u);

	// destroy library handles
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);
}

void waveD(int N, double *v, double *u, double ti){
	static cufftHandle fftPlan, ifftPlan;
	static cufftDoubleComplex *u_hat;
	// very important length(u)=length(v)=N
	int n=N/2;
	if(!u_hat){
		cufftPlan1d(&fftPlan,  n, CUFFT_D2Z, 1);
		cufftPlan1d(&ifftPlan, n, CUFFT_Z2D, 1);
		cudaMalloc((void**)&u_hat, n/2*sizeof(cufftDoubleComplex));
	}
	fftD(fftPlan, ifftPlan, n, 1, v, u+n, u_hat);
	fftD(fftPlan, ifftPlan, n, 1, v+n, u, u_hat);
}

void waveExample(int N){
	// Solves the one-dimensional wave equation u_xx = u_tt, given initial u(x) and du/dt

	// Initial conditions
	double *u;
	cudaMalloc((void**)&u, 2*N*sizeof(double));
	auto f1=[] __device__ (double x){
		double x1=x-pi/2;
		double x2=x+pi/2;
		return exp(-40*x1*x1)+exp(-40*x2*x2);
	};
	cudaMap(f1, N, u, -pi, pi-2*pi/N);
	auto f2=[] __device__ (double x){
		double x1=x-pi/2;
		double x2=x+pi/2;
		return exp(-40*x1*x1)-exp(-40*x2*x2);
	};
	cudaMap(f2, N, u+N, -pi, pi-2*pi/N);

	cublasHandle_t cublasH;
	cublasCreate(&cublasH);
	RungeKutta wave(cublasH, 2*N, u, 0.0, waveD);

	uchar4 *rgba;
	cudaMalloc((void**)&rgba, N*N*sizeof(uchar4));
	double umin=0, umax=2;
	//minmax(&umin, &umax, N, u);
	auto plot=[umin, umax] __device__ (double u){
		return hot((float)((u-umin)/(umax-umin)));
	};

	double dt=pi/N;
	int nframes=N;
	for(int i=0; i<nframes; i++){
		wave.solve(dt);
		cudaMap(plot, N, u, rgba+i*N);
	}
	string path="/home/pbrubeck/ParallelUniverse/cuMatlab/data/wave.png";
	imwrite(N, N, rgba, path);
	cublasDestroy(cublasH);
}

void cufftExample(int N){
	// Computes first derivative of a periodic function f(x) on [-pi, pi]
	double *u;
	cudaMallocManaged((void**)&u, N*sizeof(double));
	auto f=[]__device__(double th)->double{return sin(th);};
	double dx=2*pi/N;
	cudaMap(f, N, u, dx, dx*N);

	cufftHandle fftPlan, ifftPlan;
	cufftPlan1d(&fftPlan,  N, CUFFT_D2Z, 1);
	cufftPlan1d(&ifftPlan, N, CUFFT_Z2D, 1);

	cufftDoubleComplex *uhat;
	cudaMalloc((void**)&uhat, N*sizeof(cufftDoubleComplex));

	fftD(fftPlan, ifftPlan, N, 1, u, u, uhat);

	cudaDeviceSynchronize();
	disp(N, u);

	cufftDestroy(fftPlan);
	cufftDestroy(ifftPlan);
}



#endif /* EXAMPLES_H_ */
