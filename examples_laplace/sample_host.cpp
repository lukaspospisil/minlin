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


double getUnixTime(void){
	struct timespec tv;
	if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;
	return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
}


HostVector<double> my_multiplication_for(HostVector<double> x){
	int N = x.size();
	HostVector<double> Ax(N);
	int t;

	for(t=0;t<N;t++){
		/* first row */
		if(t == 0){
			Ax(t) = x(t) - x(t+1);
		}
		/* common row */
		if(t > 0 && t < N-1){
			Ax(t) = -x(t-1) + 2.0*x(t) - x(t+1);
		}
		/* last row */
		if(t == N-1){
			Ax(t) = -x(t-1) + x(t);
		}
	}

    return Ax;	
}

HostVector<double> my_multiplication_omp(HostVector<double> x){
	int N = x.size();
	HostVector<double> Ax(N);
	int t;

	#pragma omp parallel for
	for(t=0;t<N;t++){
		/* first row */
		if(t == 0){
			Ax(t) = x(t) - x(t+1);
		}
		/* common row */
		if(t > 0 && t < N-1){
			Ax(t) = -x(t-1) + 2.0*x(t) - x(t+1);
		}
		/* last row */
		if(t == N-1){
			Ax(t) = -x(t-1) + x(t);
		}
	}

    return Ax;	
}



int main() {

    int N = 50000;
    int t;

    // allocate storage
    HostVector<double> x(N);
    HostMatrix<double> A(N,N);

	double t_start, t1, t2, t3; /* to measure time */

	/* fill matrix and vector */
	A(all) = 0.0;
	x(all) = 0.0;
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

    HostVector<double> Ax1;
    HostVector<double> Ax2;
    HostVector<double> Ax3;

	for(t = 0;t < 10;t++){
		t_start = getUnixTime();
		Ax1 = A*x; 
		t1 = getUnixTime() - t_start;
		
		t_start = getUnixTime();
		Ax2 = my_multiplication_for(x);
		t2 = getUnixTime() - t_start;

		t_start = getUnixTime();
		Ax3 = my_multiplication_omp(x);
		t3 = getUnixTime() - t_start;
	
		/* content */
//		std::cout << "minlin: " << Ax1 << std::endl;	
//		std::cout << "for:    " << Ax2 << std::endl;	
//		std::cout << "omp:    " << Ax3 << std::endl;	

		/* norm */
		std::cout << "minlin: " << t1 << "s, norm(Ax)_" << t << " = " << norm(Ax1) << std::endl;
		std::cout << "for:    " << t2 << "s, norm(Ax)_" << t << " = " << norm(Ax2) << std::endl;
		std::cout << "omp:    " << t3 << "s, norm(Ax)_" << t << " = " << norm(Ax3) << std::endl;

		std::cout << "-----------------------------------------------------------" << std::endl;

	}
	
}
