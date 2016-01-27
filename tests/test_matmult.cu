/**
	Let A be a Laplace (tridiagonal) matrix

    We are interested in the comparison of several ways how to compute A*x with MINLIN:
    TEST_MINLIN_FULL - create dense minlin-matrix and multiply with it
    TEST_MINLIN      - use idea from Ben: Ax = -x(..) + 2*x(..) - x(..)
    TEST_FOR         - use naive sequential "for" cycle
    TEST_OMP         - run the previous "for" cycle as OpenMP "parallel for"
    TEST_CUDA        - iteration of "for" cycle is runned as CUDA kernel

**/

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

#include <stdio.h> /* printf in cuda */
#include <limits> /* max value of double/float */

#include <omp.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

using namespace minlin::threx;

/* compute on device or host ? and which tests to run ? */
#ifdef USE_GPU
	/* compute using CUDA on Device */

	#define MyVector DeviceVector
	#define MyMatrix DeviceMatrix

	#define TEST_MINLIN_FULL false
	#define TEST_MINLIN true
	#define TEST_FOR false
	#define TEST_OMP false
	#define TEST_CUDA true

#else
	/* compute without CUDA on Host */

	#define MyVector HostVector
	#define MyMatrix HostMatrix

	#define TEST_MINLIN_FULL false
	#define TEST_MINLIN true
	#define TEST_FOR true
	#define TEST_OMP true
	#define TEST_CUDA false

#endif

/* double/float values in Vector? */
#define Scalar double

MINLIN_INIT

/* cuda error check */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"\n\x1B[31mCUDA error:\x1B[0m %s %s \x1B[33m%d\x1B[0m\n\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* timer */
double getUnixTime(void){
	struct timespec tv;
	if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;
	return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
}


/* A*x using MINLIN matrix-vector multiplication (with dense matrix) */
void my_multiplication_minlin_full(MyVector<Scalar> Ax, MyMatrix<Scalar> A, MyVector<Scalar> x){
	Ax = A*x;
}

/* A*x using MINLIN with vectors (idea from Ben) */
void my_multiplication_minlin(MyVector<Scalar> Ax, MyVector<Scalar> x){
	int N = x.size();

	Ax(1,N-2) = 2*x(1,N-2) - x(0,N-3) - x(2,N-1);
	
	/* first and last */
	Ax(0) = x(0) - x(1);
	Ax(N-1) = x(N-1) - x(N-2);
	
}


/* A*x using simple sequential "for" cycle */
void my_multiplication_for(MyVector<Scalar> Ax, MyVector<Scalar> x){
	int N = x.size();
	int t;

	for(t=0;t<N;t++){
		/* first row */
		if(t == 0){
			Ax(t) = x(t) - x(t+1);
		}
		/* common row */
		if(t > 0 && t < N-1){
			Ax(t) = -x(t-1) + 2.0*x(t) - x(t+1);
		}
		/* last row */
		if(t == N-1){
			Ax(t) = -x(t-1) + x(t);
		}
	}
}

/* A*x using OpenMP */
void my_multiplication_omp(MyVector<Scalar> Ax, MyVector<Scalar> x){
	int N = x.size();
	int t;

	int nb_thread = omp_get_thread_num();

	#pragma omp parallel for private(t)
	for(t=0;t<N;t++){
		
		/* first row */
		if(t == 0){
			Ax(t) = x(t) - x(t+1);
		}
		/* common row */
		if(t > 0 && t < N-1){
			Ax(t) = -x(t-1) + 2.0*x(t) - x(t+1);
		}
		/* last row */
		if(t == N-1){
			Ax(t) = -x(t-1) + x(t);
		}
	}
	
}

/* A*x using CUDA kernel */
template <typename T> __global__
void kernel_mult(T* Axp, T* xp, int N)
{
	/* compute my id */
	int t = blockIdx.x*blockDim.x + threadIdx.x;

	/* test access to array with vector values */
//	printf("x(%d) = %f\n",t,xp[t]);

	/* first row */
	if(t == 0){
		Axp[t] = xp[t] - xp[t+1];
	}
	/* common row */
	if(t > 0 && t < N-1){
		Axp[t] = -xp[t-1] + 2.0*xp[t] - xp[t+1];
	}
	/* last row */
	if(t == N-1){
		Axp[t] = -xp[t-1] + xp[t];
	}

	/* if t >= N then relax and do nothing */	

}

void my_multiplication_cuda(MyVector<Scalar> Ax, MyVector<Scalar> x){
	int N = x.size();

	/* call cuda kernels */
	/* pass a thrust raw pointers to cuda kernel */
	Scalar *xp = x.pointer(); /* thank minlin for this function! */
	Scalar *Axp = Ax.pointer();

	kernel_mult<<<N, 1>>>(Axp,xp,N);

	/* synchronize kernels, if there is an error with cuda, then it will appear here */ 
	gpuErrchk( cudaDeviceSynchronize() );
}

/* fill vector using CUDA kernel */
template <typename T> __global__
void fill_x(T* x, int N)
{
	/* compute my id */
	int t = blockIdx.x*blockDim.x + threadIdx.x;

	if(t < N){
		x[t] = 1.0 + 1.0/(Scalar)(t+1);;
	}
	
	/* if t >= N then relax and do nothing */	
}



int main ( int argc, char *argv[] ) {

	/* read command line arguments */
	if(argc < 2){
		std::cout << "1. argument - N - the dimension of the problem" << std::endl;
		std::cout << "2. argument - M - number of tests (default 10)" << std::endl;

		std::cout << std::endl << argv[0] << " N" << std::endl;
		std::cout << argv[0] << " N M" << std::endl;
		return 1;
	}

	int k; /* iterator */
	int N = atoi(argv[1]); /* the first argument is the dimension of problem */
	std::cout << "N = " << N << " (dimension)" << std::endl;

	int M = 10; /* default number of tests */
	if(argc >= 3){
		M = atoi(argv[2]); /* the second (optional) argument is the number of tests */
	}
	std::cout << "M = " << M << " (number of tests)" << std::endl;

	double t_start, t; /* to measure time */

	/* fill vector with some values */
	t_start = getUnixTime();
	MyVector<Scalar> x(N);
	x(all) = 0.0;
	#ifdef USE_GPU
		/* fill vector using CUDA */
		// TODO: optimal number of threads/block
		Scalar *xp = x.pointer();
		fill_x<<<N, 1>>>(xp,N);
		gpuErrchk( cudaDeviceSynchronize() );
		
	#else
		/* fill vector using OpenMP */
		#pragma omp parallel for private(k)
		for(k=0;k<N;k++){
			/* vector */
			x(k) = 1.0 + 1.0/(Scalar)(k+1);
		}	
		
	#endif
		
	std::cout << "init & fill vector: " << getUnixTime() - t_start << "s" << std::endl;


	/* if it is MINLIN_FULL test, create&fill the matrix */
	#if TEST_MINLIN_FULL
		t_start = getUnixTime();

		MyMatrix<Scalar> A(N,N);

		A(all) = 0.0;

		for(k=0;k<N;k++){
			/* first row */
			if(k == 0){
				A(k,k) = 1.0;
				A(k,k+1) = -1.0;
			}
			/* common row */
			if(k > 0 && k < N-1){
				A(k,k+1) = -1.0;
				A(k,k) = 2.0;
				A(k,k-1) = -1.0;
			}
			/* last row */
			if(k == N-1){
				A(k,k-1) = -1.0;
				A(k,k) = 1.0;
			}
		}	
		std::cout << "init & fill matrix: " << getUnixTime() - t_start << "s" << std::endl;

	#endif

	std::cout << std::endl;

	/* to compute average time of each algorithm */
	/* these variables store the sum of computing times */
	#if TEST_MINLIN_FULL
		double t_minlin_full = 0.0;
	#endif
	#if TEST_MINLIN
		double t_minlin = 0.0;
	#endif
	#if TEST_FOR
		double t_for = 0.0;
	#endif
	#if TEST_OMP
		double t_omp = 0.0;
	#endif
	#if TEST_CUDA
		double t_cuda = 0.0;
	#endif

	/* multiplication test */
	MyVector<Scalar> Ax(N);
	
	/* I want to see the problems with setting the vector values immediately in the norm */
	/* if I forget to set a component of Ax, then the norm will be huge */
	Scalar default_value = std::numeric_limits<Scalar>::max(); 

	for(k = 0;k < M;k++){ /* for every test */
		std::cout << k+1 << ". test (of " << M << ")" << std::endl;
		
		#if TEST_MINLIN_FULL
			Ax(all) = default_value; /* clean previous results */

			t_start = getUnixTime();
			my_multiplication_minlin_full(Ax, A, x);
			t = getUnixTime() - t_start;

			std::cout << " minlin_full: " << t << "s, norm(Ax) = " << norm(Ax) << std::endl;
			t_minlin_full += t;

			/* if the dimension is small, then show also the content */
			if(N <= 15) std::cout << "  " << Ax << std::endl;	
		#endif

		#if TEST_MINLIN
			Ax(all) = default_value; /* clean previous results */

			t_start = getUnixTime();
			my_multiplication_minlin(Ax, x);
			t = getUnixTime() - t_start;

			std::cout << " minlin: " << t << "s, norm(Ax) = " << norm(Ax) << std::endl;
			t_minlin += t;

			if(N <= 15) std::cout << "  " << Ax << std::endl;	
		#endif
		
		#if TEST_FOR
			Ax(all) = default_value; /* clean previous results */

			t_start = getUnixTime();
			my_multiplication_for(Ax, x);
			t = getUnixTime() - t_start;

			std::cout << " for:    " << t << "s, norm(Ax) = " << norm(Ax) << std::endl;
			t_for += t;

			if(N <= 15) std::cout << "  " << Ax << std::endl;	
		#endif
		
		#if TEST_OMP
			Ax(all) = default_value; /* clean previous results */

			t_start = getUnixTime();
			my_multiplication_omp(Ax, x);
			t = getUnixTime() - t_start;

			std::cout << " omp:    " << t << "s, norm(Ax) = " << norm(Ax) << std::endl;
			t_omp += t;

			if (N <= 15) std::cout << "  " << Ax << std::endl;

		#endif

		#if TEST_CUDA
			Ax(all) = default_value; /* clean previous results */

			t_start = getUnixTime();
			my_multiplication_cuda(Ax, x);
			t = getUnixTime() - t_start;

			std::cout << " cuda:   " << t << "s, norm(Ax) = " << norm(Ax) << std::endl;
			t_cuda += t;

			if(N <= 15) std::cout << "  " << Ax << std::endl;	
		#endif

		std::cout << "-----------------------------------------------------------" << std::endl;

	}
	
	
	
	/* give final info with average times */
	std::cout << std::endl;
	std::cout << "N = " << N << " (dimension)" << std::endl;
	std::cout << "M = " << M << " (number of tests)" << std::endl;
	std::cout << "average times:" << std::endl;

	/* compute and show average computing times */
	#if TEST_MINLIN_FULL
		std::cout << "minlin_full: " << t_minlin_full/(double)M << std::endl;
	#endif
	#if TEST_MINLIN
		std::cout << "minlin:      " << t_minlin/(double)M << std::endl;
	#endif
	#if TEST_FOR
		std::cout << "for:         " << t_for/(double)M << std::endl;
	#endif
	#if TEST_OMP
		std::cout << "omp:         " << t_omp/(double)M << std::endl;
	#endif
	#if TEST_CUDA
		std::cout << "cuda:        " << t_cuda/(double)M << std::endl;
	#endif

	
}
