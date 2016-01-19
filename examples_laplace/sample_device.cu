/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

#include <omp.h>


using namespace minlin::threx;

#define MyVector HostVector
#define MyMatrix HostMatrix
#define Scalar double

#define TEST_MINLIN true
#define TEST_FOR true
#define TEST_OMP true

MINLIN_INIT

/* timer */
double getUnixTime(void){
	struct timespec tv;
	if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;
	return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
}

/* A*x using simple for cycle */
void my_multiplication_for(MyVector<Scalar> *Ax, MyVector<Scalar> x){
	int N = x.size();
	int t;

	for(t=0;t<N;t++){
		/* first row */
		if(t == 0){
			(*Ax)(t) = x(t) - x(t+1);
		}
		/* common row */
		if(t > 0 && t < N-1){
			(*Ax)(t) = -x(t-1) + 2.0*x(t) - x(t+1);
		}
		/* last row */
		if(t == N-1){
			(*Ax)(t) = -x(t-1) + x(t);
		}
	}
}

void my_multiplication_omp(MyVector<Scalar> *Ax, MyVector<Scalar> x){
	int N = x.size();
	int t;

	#pragma omp parallel for 
	for(t=0;t<N;t++){
		/* first row */
		if(t == 0){
			(*Ax)(t) = x(t) - x(t+1);
		}
		/* common row */
		if(t > 0 && t < N-1){
			(*Ax)(t) = -x(t-1) + 2.0*x(t) - x(t+1);
		}
		/* last row */
		if(t == N-1){
			(*Ax)(t) = -x(t-1) + x(t);
		}
	}
}



int main ( int argc, char *argv[] ) {

	/* read command line arguments */
	if(argc < 2){
		std::cout << argv[0] << " N" << std::endl;
		return 1;
	}

	int N = atoi(argv[1]);	
    int t;

	double t_start, t1, t2, t3; /* to measure time */

	std::cout << "N = " << N << std::endl;

	/* fill matrix and vector */
	t_start = getUnixTime();
    MyVector<Scalar> x(N);
	x(all) = 0.0;
	for(t=0;t<N;t++){
		/* vector */
		x(t) = 1.0 + 1.0/(Scalar)(t+1);
	}	
	std::cout << "set values of vector: " << getUnixTime() - t_start << "s" << std::endl;


    #if TEST_MINLIN
		t_start = getUnixTime();

		MyMatrix<Scalar> A(N,N);

		A(all) = 0.0;

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
		}	
		std::cout << "set values of matrix: " << getUnixTime() - t_start << "s" << std::endl;

	#endif

	std::cout << std::endl;

	/* multiplication test */
    #if TEST_MINLIN
		MyVector<Scalar> Ax1(N); /* MINLIN A*x */
    #endif
    #if TEST_FOR
		MyVector<Scalar> Ax2(N); /* A*x using simple for */
    #endif
    #if TEST_OMP
		MyVector<Scalar> Ax3(N); /* A*x using omp */
	#endif

	for(t = 0;t < 10;t++){
	    #if TEST_MINLIN
			t_start = getUnixTime();
//			Ax1 = A*x; 
			Ax1(all) = A*x; 

			t1 = getUnixTime() - t_start;
			std::cout << "minlin: " << t1 << "s, norm(Ax)_" << t << " = " << norm(Ax1) << std::endl;
		#endif
		
		#if TEST_FOR
			t_start = getUnixTime();
			my_multiplication_for(&Ax2,x);
			t2 = getUnixTime() - t_start;
			std::cout << "for:    " << t2 << "s, norm(Ax)_" << t << " = " << norm(Ax2) << std::endl;
		#endif
		
		#if TEST_OMP
			t_start = getUnixTime();
			my_multiplication_omp(&Ax3,x);
			t3 = getUnixTime() - t_start;
			std::cout << "omp:    " << t3 << "s, norm(Ax)_" << t << " = " << norm(Ax3) << std::endl;
		#endif
	
		/* content */
		if(N <= 15){
			#if TEST_MINLIN
				std::cout << "minlin: " << Ax1 << std::endl;	
			#endif
			
			#if TEST_FOR
				std::cout << "for:    " << Ax2 << std::endl;	
			#endif
			
			#if TEST_OMP
				std::cout << "omp:    " << Ax3 << std::endl;	
			#endif	
		}

		std::cout << "-----------------------------------------------------------" << std::endl;

	}
	
}
