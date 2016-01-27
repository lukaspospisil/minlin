/**
	test how large vectors could be stored in memory of CPU/GPU

**/

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

#include <stdio.h> /* printf in cuda */
#include <stdlib.h> /* atoi, strtol */
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

	#define TEST_CPU true
	#define TEST_GPU true

#else
	/* compute without CUDA on Host */

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
	long int N = strtol(argv[1],NULL,0); /* the first argument is the dimension of problem */
	std::cout << "N = " << N << " (dimension)" << std::endl;

	int M = 10; /* default number of tests */
	if(argc >= 3){
		M = atoi(argv[2]); /* the second (optional) argument is the number of tests */
	}
	std::cout << "M = " << M << " (number of tests)" << std::endl;

	double t_start, t; /* to measure time */


	/* to compute average time of each algorithm */
	/* these variables store the sum of computing times */
	#if TEST_CPU
		double t_cpu = 0.0;
		HostVector<Scalar> *x_cpu;
		x_cpu = new HostVector<Scalar>(N);
		delete x_cpu;
	#endif
	#if TEST_GPU
		double t_gpu = 0.0;
		DeviceVector<Scalar> *x_gpu;
		x_gpu = new DeviceVector<Scalar>(N);
		delete x_gpu;
	#endif


	for(k = 0;k < M;k++){ /* for every test */
		std::cout << k+1 << ". test (of " << M << ")" << std::endl;
		
		#if TEST_CPU

			t_start = getUnixTime();
			x_cpu = new HostVector<Scalar>(N);
			(*x_cpu)(all) = 1.0;
			t = getUnixTime() - t_start;

			std::cout << " cpu: " << t << "s, norm(x) = " << norm(*x_cpu) << ", size = " << (*x_cpu).size()*sizeof(Scalar) << std::endl;
			t_cpu += t;

			/* if the dimension is small, then show also the content */
			if(N <= 15) std::cout << "x_cpu:" << "  " << *x_cpu << std::endl;	

			delete x_cpu;
		#endif

		#if TEST_GPU

			t_start = getUnixTime();
			x_gpu = new DeviceVector<Scalar>(N);
			(*x_gpu)(all) = 1.0;
			t = getUnixTime() - t_start;

			std::cout << " gpu: " << t << "s, norm(x) = " << norm(*x_gpu) << ", size = " << (*x_gpu).size()*sizeof(Scalar) << std::endl;
			t_gpu += t;

			/* if the dimension is small, then show also the content */
			if(N <= 15) std::cout << "x_gpu:" << "  " << *x_gpu << std::endl;	
			
			delete x_gpu;
		#endif

		std::cout << "-----------------------------------------------------------" << std::endl;

	}
	
	
	
	/* give final info with average times */
	std::cout << std::endl;
	std::cout << "N = " << N << " (dimension)" << std::endl;
	std::cout << "M = " << M << " (number of tests)" << std::endl;
	std::cout << "average times:" << std::endl;

	/* compute and show average computing times */
	#if TEST_CPU
		std::cout << "cpu: " << t_cpu/(double)M << std::endl;
	#endif
	#if TEST_GPU
		std::cout << "gpu: " << t_gpu/(double)M << std::endl;
	#endif

	
}
