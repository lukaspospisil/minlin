/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

#include <stdio.h> /* printf in cuda */
#include <limits> /* max value of double/float */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

using namespace minlin::threx;

/* compute on device or host ? */
#ifdef USE_GPU
	#define MyVector DeviceVector
#else
	#define MyVector HostVector
#endif

#define Scalar double

MINLIN_INIT

/* timer */
double getUnixTime(void){
	struct timespec tv;
	if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;
	return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
}

void my_sort(MyVector<Scalar> *x){
	std::cout << "thrust fun" << std::endl;
	int k;
	
	thrust::host_vector<Scalar> y((*x).size());
	for(k=0;k<(*x).size();k++){
		y[k] = (*x)(k);
	}

	thrust::sort(y.begin(), y.end());

	for(k=0;k<(*x).size();k++){
		(*x)(k) = y[k];
	}
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

	int M = 10; /* number of tests */
	if(argc >= 3){
		M = atoi(argv[2]); /* the second (optional) argument is the number of tests */
	}
	std::cout << "M = " << M << " (number of tests)" << std::endl;

	double t_start, t; /* to measure time */

	/* fill vector with some values */
	t_start = getUnixTime();
    MyVector<Scalar> x(N);
	x(all) = 0.0;
	for(k=0;k<N;k++){
		/* vector */
		x(k) =  rand()%100;
	}	
	std::cout << "init & fill vector: " << getUnixTime() - t_start << "s" << std::endl;
	std::cout << std::endl;

	double t_sort = 0.0;

	/* I want to keep original vector x */
    MyVector<Scalar> x_sorted(N);
	
	for(k = 0;k < M;k++){
		std::cout << k+1 << ". test (of " << M << ")" << std::endl;
		
		x_sorted = x;

		t_start = getUnixTime();
		my_sort(&x_sorted);
		t = getUnixTime() - t_start;

		std::cout << " sorted in: " << t << "s" << std::endl;
		t_sort += t;

		if(N <= 15) std::cout << "  " << x_sorted << std::endl;	

		std::cout << "-----------------------------------------------------------" << std::endl;

	}
	
	
	
	/* give final info with average times */
	std::cout << std::endl;
	std::cout << "N = " << N << " (dimension)" << std::endl;
	std::cout << "M = " << M << " (number of tests)" << std::endl;
	std::cout << "average time:" << t_sort/(double)M << std::endl;
	
}
