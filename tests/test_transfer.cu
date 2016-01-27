/**
	test the speed of transfering a MinLin vector from Host to Device

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

#ifdef USE_GPU
	/* compute using CUDA on Device */

	#define MyVector1 HostVector
	#define MyVector2 DeviceVector

	/* for cout informations */
	#define MyName1 "CPU"
	#define MyName2 "GPU"
	
#else
	/* compute without CUDA on Host - this should be quick since there is not communication Host-Device */

	#define MyVector1 HostVector
	#define MyVector2 HostVector

	/* for cout informations */
	#define MyName1 "CPU"
	#define MyName2 "CPU"

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
	double t_1to2 = 0.0;
	double t_2to1 = 0.0;

	MyVector1<Scalar> x1(N);
	MyVector2<Scalar> x2(N);

	for(k = 0;k < M;k++){ /* for every test */
		std::cout << k+1 << ". test (of " << M << ")" << std::endl;

		/* 1 to 2 */	
		x1(all) = 1.0;
		x2(all) = 2.0;
		t_start = getUnixTime();
		x2 = x1;
		t = getUnixTime() - t_start;

		std::cout << MyName1 << " to " << MyName2 << ": " << t << std::endl;
		t_1to2 += t;

		/* if the dimension is small, then show also the content */
		if(N <= 15){
			std::cout << "x1(" << MyName1 << "): " << x1 << std::endl;	
			std::cout << "x2(" << MyName2 << "): " << x2 << std::endl;	
		}

		/* 2 to 1 */	
		x1(all) = 1.0;
		x2(all) = 2.0;
		t_start = getUnixTime();
		x1 = x2;
		t = getUnixTime() - t_start;

		std::cout << MyName2 << " to " << MyName1 << ": " << t << std::endl;
		t_2to1 += t;

		if(N <= 15){
			std::cout << "x1(" << MyName1 << "): " << x1 << std::endl;	
			std::cout << "x2(" << MyName2 << "): " << x2 << std::endl;	
		}

		std::cout << "-----------------------------------------------------------" << std::endl;
	}
	
	
	
	/* give final info with average times */
	std::cout << std::endl;
	std::cout << "N = " << N << " (dimension)" << std::endl;
	std::cout << "M = " << M << " (number of tests)" << std::endl;
	std::cout << "average times:" << std::endl;

	/* compute and show average computing times */
	std::cout << " " << MyName1 << " to " << MyName2 << ": " << t_1to2/(double)M << std::endl;
	std::cout << " " << MyName2 << " to " << MyName1 << ": " << t_2to1/(double)M << std::endl;

	
}
