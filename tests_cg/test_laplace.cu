
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

/* my petsc stuff */
#include "petscvector.h"
#include "laplacefullmatrix.h"
#include "cg.h"


using namespace minlin::threx;
MINLIN_INIT


int DEBUG_MODE = 0;
bool PETSC_INITIALIZED = false;

//typedef HostVector<double> HostVectorDouble;

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

		std::cout << std::endl << argv[0] << " N" << std::endl;
		return 1;
	}

	int N = atoi(argv[1]); /* the first argument is the dimension of problem */
	std::cout << "N = " << N << " (dimension)" << std::endl;

	double t_start, t; /* to measure time */


	/* -------------------------------- PETSC TEST -------------------------*/

	std::cout << "-------------------------------- PETSC TEST -------------------------" << std::endl;

	/* initialize Petsc */
	PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
	PETSC_INITIALIZED = true;
	
	/* create vectors */
	PetscVector x0(N); /* create basic layout */
	x0(all) = 0.0;

	PetscVector b(x0); /* initialize, duplicate layout of x0 */
	b(all) = 1.0;
	b(0) = 0.0;
	b(N-1) = 0.0;

	LaplaceFullMatrix<PetscVector> A(x0); /* prepare laplace matrix */

	PetscVector x; /* solution */

	/* do some fun */
	t_start = getUnixTime();

	x = cg(A, b, x0);

	t = getUnixTime() - t_start;

	/* sufficiently small problem - give info */
	if(N <= 10){
		std::cout << "A: " << A << std::endl;
		std::cout << "x0: " << x0 << std::endl;
		std::cout << "b: " << b << std::endl;
		std::cout << "x: " << x << std::endl;
	}
	std::cout << "time: " << t << "s" << std::endl;

	PETSC_INITIALIZED = false;
	PetscFinalize();


	/* -------------------------------- MINLINHOST TEST -------------------------*/

	std::cout << "-------------------------------- MINLINHOST TEST -------------------------" << std::endl;

	/* create vectors */
	HostVectorD x0m(N);
	x0m(all) = 0.0;

	HostVectorD bm(N);
	bm(all) = 1.0;
	bm(0) = 0.0;
	bm(N-1) = 0.0;

	LaplaceFullMatrix<HostVectorD> Am(x0m); /* prepare laplace matrix */

	HostVectorD xm(N); /* solution */

	/* do some fun */
	t_start = getUnixTime();

	xm = cg(Am, bm, x0m);

	t = getUnixTime() - t_start;

	/* sufficiently small problem - give info */
	if(N <= 10){
		std::cout << "A: " << Am << std::endl;
		std::cout << "x0: " << x0m << std::endl;
		std::cout << "b: " << bm << std::endl;
		std::cout << "x: " << xm << std::endl;
	}
	std::cout << "time: " << t << "s" << std::endl;

	
}
