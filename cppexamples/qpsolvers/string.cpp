/*******************************************************************************
Solution of string problem

*******************************************************************************/
#include <thrust/functional.h>


#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

using namespace minlin::threx;


int main(int argc, char *argv[]) {
    typedef double real; /* we are going to compute in double/float? */ 
	int i; /* iterator */
    int N = 10; /* number of nodes - dimension of problem */
	real h = 1.0/(N-1); /* discretization */
	real my_eps = 0.0001; /* precision */

	/* get N from console */
	if ( argc == 2 ){
		N = atoi(argv[1]);
		h = 1.0/(N-1);
	}

    /* allocate storage */
    HostMatrix<real> A(N, N); /* Hessian matrix */
    HostVector<real> b(N); /* linear term */
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

	/* scale to 1/h */
	A = (1.0/h)*A;
	b = h*b;
//	A = A;
//	b = h*h*b;

	/* print problem */
    std::cout << "A:" << std::endl;
    std::cout << A << std::endl << std::endl;
    std::cout << "b:" << std::endl;
    std::cout << b << std::endl << std::endl;

	/* CG method */
	HostVector<real> g(N); /* gradient */
	HostVector<real> p(N); /* A-conjugate vector */
	HostVector<real> Ap(N); /* A*p */
	int it = 0; /* iteration counter */
	real normb, normg, alpha, beta, pAp, gg, gg_old;
	
	x(minlin::all) = 0.0; /* initial approximation */  
	g -= b; /* compute gradient */
	p = g; /* initial conjugate gradient */

	normb = norm(b);
	gg = dot(g,g);
	normg = sqrt(gg);
	while(gg > my_eps*normb && it < 10000){
		/* compute new approximation */
/*		Ap = A*p;
		pAp = dot(Ap,p);
		alpha = gg/pAp;
		x -= alpha*p;
*/		
//		g = A*x; g -= b; /* compute gradient */
/*		g -= alpha*Ap;
		gg_old = gg;
		gg = dot(g,g);
		normg = sqrt(gg);
*/			
		/* compute new A-orthogonal vector */
/*		beta = gg/gg_old;
		p = beta*p;
		p += g;
*/		

		/* SD */
		Ap = A*g;
		pAp = dot(Ap,g);
		alpha = gg/pAp;

		std::cout << "g:" << std::endl;
		std::cout << g << std::endl << std::endl;
		std::cout << "Ag:" << std::endl;
		std::cout << Ap << std::endl << std::endl;


		std::cout << "it " << it << ": ||g|| = " << normg << ", ||g||/||b|| = " << normg/normb << std::endl;
		std::cout << "       gg = " << gg << ", gAg = " << pAp << ", alpha = " << alpha << std::endl;

		x -= alpha*g;
		g = A*x; g -= b; /* compute gradient */
		gg = dot(g,g);
		normg = sqrt(gg);		
				
		it += 1;


	}


	/* print solution */
    std::cout << "x:" << std::endl;
    std::cout << x << std::endl << std::endl;


}
