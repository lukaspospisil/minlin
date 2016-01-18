/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

using namespace minlin::threx;

MINLIN_INIT

int main() {

    int N = 500;
    int t;

    // allocate storage
    HostVector<double> x(N);
    HostMatrix<double> A(N,N);

	for(t=0;t<N;t++){
		/* first row */
		if(t == 0){
			A(t,t) = 1.0;
			A(t,t+1) = -1.0;
		}
		/* common row */
		if(t > 0 && t < N-1){
			A(t,t+1) = -1.0;
			A(t,t) = 2.0;
			A(t,t-1) = -1.0;
		}
		/* last row */
		if(t == N-1){
			A(t,t-1) = -1.0;
			A(t,t) = 1.0;
		}
		
		/* vector */
		x(t) = 1.0 + 1.0/(double)(t+1);
	}	

//	std::cout << "A = " << A << std::endl;
//	std::cout << "x = " << x << std::endl;

    HostVector<double> Ax;

	for(t = 0;t < 10;t++){
		Ax = A*x;
		std::cout << "norm(Ax)_" << t << " = " << norm(Ax) << std::endl;
		
	}
	
	
}
