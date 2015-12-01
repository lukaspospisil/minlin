/*******************************************************************************
Solution of string problem with QPOPT library

*******************************************************************************/
//#define MINLIN_DEBUG
#define QPOPT_DEBUG
//#define QPOPT_DEBUG2
#define QPOPT_DEBUG_F

#include <thrust/functional.h>

#include <iostream>
#include <fstream>

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
	real my_eps = 0.001; /* precision */
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
    HostMatrix<real> B(2, N); /* equality constraints */

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
			l(i) = -0.5;
		} else {
			l(i) = -0.7;
		}

		/* force */
		b(i) = -15; 
    }
//	A(0,0) = 1.0;
//	A(N-1,N-1) = 1.0;

	/* Dirichlet boundary condition */
	B(0,0) = 1.0;
	B(1,N-1) = 1.0;

	/* scale to [0,1] */
	A = (1.0/h)*A;
	b = h*b;
//	A = A;
//	b = h*h*b;


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

//	normA = (1.0/h)*4.0;

	normA = (1.0/h)*4.0;
//	x = minlin::QPOpt::solve_bound(A,normA,b,l,my_eps);
//	x = minlin::QPOpt::solve_unconstrained(A,b,my_eps);
	x = minlin::QPOpt::solve_eqbound(A,normA,b,l,B,normBTB,my_eps);


	/* save solution */
	char name_of_file[256];					/* the name of output VTK file */

	sprintf(name_of_file, "output_bound_%dt.vtk",NT);

	std::ofstream myfile;
	myfile.open(name_of_file);
	myfile << "# vtk DataFile Version 3.1" << std::endl;
	myfile << "this is the solution of our problem" << std::endl;
	myfile << "ASCII" << std::endl;
	myfile << "DATASET POLYDATA" << std::endl;

	/* points - coordinates */
	myfile << "POINTS " << N << " FLOAT" << std::endl;
	for(i=0;i < N;i++){
		myfile << i*h << " " << x(i) << " 0.0" << std::endl;
	}
	
	/* line solution */
	myfile << "LINES 1 " << N+1 << std::endl;
	myfile << N << " ";
	for(i=0;i < N;i++){
		myfile << i << " ";
	}
	myfile << std::endl;
	
	/* values is points */
	myfile << "POINT_DATA " << N  << std::endl;
	myfile << "SCALARS solution float 1"  << std::endl;
	myfile << "LOOKUP_TABLE default"  << std::endl;
	for(i=0;i < N;i++){
//		myfile << x(i) << std::endl;
		myfile << 1.0 << std::endl;
	}



	myfile.close();

	return 0;
}
