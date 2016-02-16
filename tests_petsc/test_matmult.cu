/**
	Let A be a Laplace (tridiagonal) matrix

    We are interested in the comparison of several ways how to compute A*x with MINLIN:
    TEST_MINLIN      - use idea from Ben: Ax = -x(..) + 2*x(..) - x(..)
    TEST_PETSC         - use naive sequential "for" cycle

**/

/* cout */
#include <iostream>

/* max value of double/float */
#include <limits> 

/* openmpi */
#include <omp.h>

/* minlin */
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

/* petsc */
#include "petsc.h"
#include "petscvector.h"


using namespace minlin::threx;
MINLIN_INIT


int DEBUG_MODE = 0;
bool PETSC_INITIALIZED = false;

#define TEST_MINLIN true /* just for control on one CPU */
#define TEST_PETSC false /* use standart Vec from Petsc, assemble dense Mat and multiply with it using standart Petsc fuctions */
#define TEST_PETSCVECTOR true /* use my minlin-matlab-style wrapper & Ben multiplication idea (I dont have a wrapper for matmult yet) */


/* double/float values in Vector? */
#define Scalar double

/* timer */
double getUnixTime(void){
	struct timespec tv;
	if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;
	return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
}


/* A*x using minlin-matlab-style with vectors (idea from Ben) */
/* will be used for manipulation with HostVector, (DeviceVector), PetscVector */
template<class VectorType>
void my_multiplication_matrixfree(VectorType &Ax, const VectorType &x){
	int N = x.size();

	Ax(1,N-2) = 2*x(1,N-2) - x(0,N-3) - x(2,N-1);
	
	/* begin and end */
	Ax(0) = x(0) - x(1);
	Ax(N-1) = x(N-1) - x(N-2);
	
}




int main ( int argc, char *argv[] ) {
	/* initialize Petsc */
	PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
	PETSC_INITIALIZED = true;


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
	#if TEST_MINLIN
		std::cout << std::endl << "MINLIN:" << std::endl;

		t_start = getUnixTime();

		/* init minlin HostVector */
		std::cout << " - init vector" << std::endl;
		HostVector<Scalar> x_minlin(N);
//		x_minlin(all) = 0.0;

		/* fill vector using OpenMP */
		std::cout << " - fill vector" << std::endl;
		#pragma omp parallel for private(k)
		for(k=0;k<N;k++){
			/* vector */
			x_minlin(k) = 1.0 + 1.0/(Scalar)(k+1);
		}	

		std::cout << " - time init & fill vector: " << getUnixTime() - t_start << "s" << std::endl;

		/* prepare result vector */
		HostVector<Scalar> Ax_minlin(N);
		
	#endif

	/* fill petsc vector with some values and prepare petsc matrix */
	#if TEST_PETSC
			
		std::cout << std::endl << "PETSC:" << std::endl;

		t_start = getUnixTime();

		/* init minlin HostVector */
		std::cout << " - init vector" << std::endl;
		Vec x_petsc;

		/* fill vector using Petsc */
		

		std::cout << " - time init & fill vector: " << getUnixTime() - t_start << "s" << std::endl;

		/* prepare result vector */
//		HostVector<Scalar> Ax_minlin(N);
		
	#endif

	/* fill vector with some values */
	#if TEST_PETSCVECTOR
		std::cout << std::endl << "PETSCVECTOR:" << std::endl;

		t_start = getUnixTime();

		/* init PetscVector */
		std::cout << " - init vector" << std::endl;
		PetscVector x_petscvector(N);
//		x_petscvector(all) = 0.0;

		/* fill vector using OpenMP */
		std::cout << " - fill vector" << std::endl;

		for(k=0;k<N;k++){
			/* vector */
			x_petscvector(k) = 1.0 + 1.0/(Scalar)(k+1);
		}	

		std::cout << " - time init & fill vector: " << getUnixTime() - t_start << "s" << std::endl;

		if(N <= 15) std::cout << "  " << x_petscvector << std::endl;	

		/* prepare result vector */
		PetscVector Ax_petscvector(N);
		
	#endif

	std::cout << std::endl;

	/* to compute average time of each algorithm */
	/* these variables store the sum of computing times */
	#if TEST_MINLIN
		double t_minlin = 0.0;
	#endif
	#if TEST_PETSC
		double t_petsc = 0.0;
	#endif
	#if TEST_PETSCVECTOR
		double t_petscvector = 0.0;
	#endif

	
	/* I want to see the problems with setting the vector values immediately in the norm */
	/* if I forget to set a component of Ax, then the norm will be huge */
	Scalar default_value = std::numeric_limits<Scalar>::max(); 

	for(k = 0;k < M;k++){ /* for every test */
		std::cout << k+1 << ". test (of " << M << ")" << std::endl;
		
		#if TEST_MINLIN
			Ax_minlin(all) = default_value; /* clean previous results */

			t_start = getUnixTime();
			my_multiplication_matrixfree(Ax_minlin,x_minlin);
			t = getUnixTime() - t_start;

			std::cout << " minlin: " << t << "s, norm(Ax) = " << norm(Ax_minlin) << std::endl;
			t_minlin += t;

			/* if the dimension is small, then show also the content */
			if(N <= 15) std::cout << "  " << Ax_minlin << std::endl;	
		#endif




		#if TEST_PETSCVECTOR
			Ax_petscvector(all) = default_value; /* clean previous results */

			t_start = getUnixTime();
			my_multiplication_matrixfree(Ax_petscvector,x_petscvector);
			t = getUnixTime() - t_start;

			std::cout << " petscvector: " << t << "s, norm(Ax) = " << norm(Ax_petscvector) << std::endl;
			t_petscvector += t;

			/* if the dimension is small, then show also the content */
			if(N <= 15) std::cout << "  " << Ax_petscvector << std::endl;	
		#endif

		std::cout << "-----------------------------------------------------------" << std::endl;

	}
	
	
	
	/* give final info with average times */
	std::cout << std::endl;
	std::cout << "N = " << N << " (dimension)" << std::endl;
	std::cout << "M = " << M << " (number of tests)" << std::endl;
	std::cout << "average times:" << std::endl;

	/* compute and show average computing times */
	#if TEST_MINLIN
		std::cout << "minlin:        " << t_minlin/(double)M << std::endl;
	#endif
	#if TEST_PETSC
		std::cout << "petsc:         " << t_petsc/(double)M << std::endl;
	#endif
	#if TEST_PETSCVECTOR
		std::cout << "petscvector:   " << t_petscvector/(double)M << std::endl;
	#endif


	PETSC_INITIALIZED = false;
	PetscFinalize();
	
}
