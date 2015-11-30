/*******************************************************************************
Solution of string problem with QPOPT library

*******************************************************************************/
//#define MINLIN_DEBUG
#define QPOPT_DEBUG
//#define QPOPT_DEBUG2
#define QPOPT_DEBUG_F

#include <thrust/functional.h>

#include <iostream>

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <qpopt.h>


using namespace minlin::threx;


int main(int argc, char *argv[]) {
    typedef double real; /* we are going to compute in double/float? */ 
	int i; /* iterator */
    int N = 10; /* number of nodes - dimension of problem */
    int NT = 1; /* number of threads */
	real h = 1.0/(N-1); /* discretization */
	real my_eps = 0.0001; /* precision */
	real normA; /* norm of hessian matrix */

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
        A(i,i) = 2.0;
        if(i>0){
			A(i,i-1) = -1.0;
		}
        if(i<N-1){
			A(i,i+1) = -1.0;
		}

		b(i) = -15.0; /* force */
    }
	A(0,0) = 1.0;
	A(N-1,N-1) = 1.0;

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
//	A = A;
//	b = h*h*b;


	/* bound constraints - nonpenetration */
	l(minlin::all) = -0.25;

	/* print problem */
	#ifdef MINLIN_DEBUG
		std::cout << "A:" << std::endl;
		std::cout << A << std::endl << std::endl;
		std::cout << "b:" << std::endl;
		std::cout << b << std::endl << std::endl;
		std::cout << "l:" << std::endl;
		std::cout << l << std::endl << std::endl;
	#endif

	normA = (1.0/h)*4.0;
//	normA = 4.0;
	x = minlin::QPOpt::solve_bound(A,normA,b,l,my_eps);
//	x = minlin::QPOpt::solve_unconstrained(A,b,my_eps);

	/* print solution */
//	#ifdef MINLIN_DEBUG
		std::cout << "x:" << std::endl;
		std::cout << x << std::endl << std::endl;
//	#endif

}
