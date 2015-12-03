/*******************************************************************************
Solution of string problem with QPOPT library

*******************************************************************************/
//#define MINLIN_DEBUG
//#define QPOPT_DEBUG
//#define QPOPT_DEBUG2
//#define QPOPT_DEBUG_F

#include <thrust/functional.h>

#include <iostream>
#include <fstream>

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <qpopt.h>
#include "savevtk.h"

using namespace minlin::threx;


int main(int argc, char *argv[]) {
    typedef double real; /* we are going to compute in double/float? */ 
	int i; /* iterator */
    int N = 10; /* number of nodes - dimension of problem */
    int NT = 1; /* number of threads */
	real h = 1.0/(N-1); /* discretization */

	/* get NT,N from console */
	if ( argc > 1 ){
		NT = atoi(argv[1]);
	}
	if ( argc > 2 ){
		N = atoi(argv[2]);
		h = 1.0/(N-1);
	}

	mkl_set_dynamic(0);
	mkl_set_num_threads(NT);
	omp_set_nested(1);

    /* allocate storage */
    HostMatrix<real> A(N, N); /* Hessian matrix */
    HostVector<real> b(N); /* linear term */
    HostVector<real> l(N); /* bound constraints */
	HostVector<real> x(N); /* solution */

	/* fill matrix */
    for (i = 0; i < N; i++) {
        /* stiffness matrix */
        A(i,i) = 2.0;
        if(i>0){
			A(i,i-1) = -1.0;
		}
        if(i<N-1){
			A(i,i+1) = -1.0;
		}

		/* bound constraint - the obstacle */
		if(i < 0.5*N){
			l(i) = -0.3;
		} else {
			l(i) = -0.5;
		}

		/* force */
		b(i) = -5.0; 
    }

	/* Dirichlet boundary condition */
	A(0,0) = 1.0;
	A(N-1,N-1) = 1.0;
	A(0,1) = 0.0;
	A(1,0) = 0.0;
	A(N-1,N-2) = 0.0;
	A(N-2,N-1) = 0.0;
	b(0) = 0.0;
	b(N-1) = 0.0;

	/* scale to [0,1] */
	A = (1.0/h)*A;
	b = h*b;

	/* print problem */
	#ifdef MINLIN_DEBUG
		std::cout << "A:" << std::endl;
		std::cout << A << std::endl << std::endl;
		std::cout << "b:" << std::endl;
		std::cout << b << std::endl << std::endl;
		std::cout << "l:" << std::endl;
		std::cout << l << std::endl << std::endl;
	#endif


	minlin::QPOpt::QPSettings settings;
	minlin::QPOpt::QPSettings_default(&settings);

	settings.norm_A = (1.0/h)*4.0;

	minlin::QPOpt::QPSettings_starttimer(&settings);
	x = minlin::QPOpt::solve_bound(&settings,A,b,l);
	minlin::QPOpt::QPSettings_stoptimer(&settings);

	/* print info about algorithm performace */
	minlin::QPOpt::QPSettings_print(settings);

	/* save solution */
	char name_of_file[256];					/* the name of output VTK file */
	sprintf(name_of_file, "output_bound_t%d_n%d.vtk",NT,N);
	savevtk(name_of_file,x);



	return 0;
}
