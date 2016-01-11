/*******************************************************************************
Solution of string problem with QPOPT library

*******************************************************************************/
//#define MINLIN_DEBUG
//#define QPOPT_DEBUG
//#define QPOPT_DEBUG2
//#define QPOPT_DEBUG_F

#include <iostream>
#include <fstream>

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>
#include <qpopt/smalbe.h>

#include "savevtk.h"

using namespace minlin::threx;

MINLIN_INIT

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
    DeviceMatrix<real> A(N, N); /* Hessian matrix */
    DeviceVector<real> b(N); /* linear term */
    DeviceVector<real> l(N); /* bound constraints */
	DeviceVector<real> x(N); /* solution */
    DeviceMatrix<real> B(2, N); /* equality constraints */

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
	B(0,0) = 1.0;
	B(1,N-1) = 1.0;

	/* scale to [0,1] */
	A = A;
	b = h*h*b;

	/* print problem */
	#ifdef MINLIN_DEBUG
		std::cout << "A:" << std::endl;
		std::cout << A << std::endl << std::endl;
		std::cout << "b:" << std::endl;
		std::cout << b << std::endl << std::endl;
		std::cout << "l:" << std::endl;
		std::cout << l << std::endl << std::endl;
		std::cout << "B:" << std::endl;
		std::cout << B << std::endl << std::endl;
	#endif

	minlin::QPOpt::QPSettings settings;
	minlin::QPOpt::QPSettings_default(&settings);

	settings.norm_A = 4.0;
	settings.norm_BTB = 1.0;
	settings.my_eps = h*0.1;

	std::cout << "h = " << h << std::endl;
	std::cout << "eps = " << settings.my_eps << std::endl;


	minlin::QPOpt::QPSettings_starttimer(&settings);
	x = minlin::QPOpt::smalbe(&settings, A, b,l,B);
	minlin::QPOpt::QPSettings_stoptimer(&settings);

	/* print info about algorithm performace */
	minlin::QPOpt::QPSettings_print(settings);

	/* save solution */
	char name_of_file[256];					/* the name of output VTK file */
	sprintf(name_of_file, "output_eq_t%d_n%d.vtk",NT,N);
	savevtk(name_of_file,x);
	
	return 0;
}
