/**
	test how large vectors could be stored in memory of CPU/GPU

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

	#define TEST_CPU false
	#define TEST_GPU true

#else
	/* compute without CUDA on Host */

	#define MyVector HostVector

	#define TEST_CPU true
	#define TEST_GPU false

#endif

/* double/float values in Vector? */
#define Scalar double

MINLIN_INIT


/* timer */
double getUnixTime(void){
	struct timespec tv;
	if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;
	return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
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
