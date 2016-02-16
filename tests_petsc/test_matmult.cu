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
#include "generalmatrix.h"


using namespace minlin::threx;
MINLIN_INIT


int DEBUG_MODE = 0;
bool PETSC_INITIALIZED = false;

#define TEST_MINLIN false /* just for control on one CPU */
#define TEST_PETSC false /* use standart Vec from Petsc, assemble dense Mat and multiply with it using standart Petsc fuctions */
#define TEST_PETSCVECTOR false /* use my minlin-matlab-style wrapper & Ben multiplication idea */
#define TEST_GENERALMATRIX_NONFREE true /* use my minlin-matlab-style wrapper */


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



/* here put arbitrary function for setting the values of test vector */
double get_some_value(int index){
	return 1.0 + 1.0/(double)(index+1);
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
		HostVector<double> x_minlin(N);
//		x_minlin(all) = 0.0;

		/* fill vector using OpenMP */
		std::cout << " - fill vector" << std::endl;
		for(k=0;k<N;k++){
			/* vector */
			x_minlin(k) = get_some_value(k);
		}	

		std::cout << " - time init & fill vector: " << getUnixTime() - t_start << "s" << std::endl;

		/* if the vector is small, then print it */
		if(N <= 15) std::cout << "  " << x_minlin << std::endl;	

		/* prepare result vector */
		HostVector<double> Ax_minlin(N);
		
	#endif

	/* fill petsc vector with some values and prepare petsc matrix */
	#if TEST_PETSC
		PetscErrorCode ierr; /* I will store & control errors using this variable */
			
		std::cout << std::endl << "PETSC:" << std::endl;

		t_start = getUnixTime();


		/* init Petsc matrix */
		std::cout << " - init matrix" << std::endl;
		Mat A_petsc;

		ierr = MatCreate(PETSC_COMM_WORLD,&A_petsc); CHKERRQ(ierr);
		ierr = MatSetSizes(A_petsc,PETSC_DECIDE,PETSC_DECIDE,N,N); CHKERRQ(ierr);
		ierr = MatSetFromOptions(A_petsc); CHKERRQ(ierr); 
//		ierr = MatSetType(A_petsc,MATMPIAIJ); CHKERRQ(ierr); 
		ierr = MatMPIAIJSetPreallocation(A_petsc,5,NULL,5,NULL); CHKERRQ(ierr); 
		ierr = MatSeqAIJSetPreallocation(A_petsc,5,NULL); CHKERRQ(ierr);

		int row,col;
		double new_value;
		for(row=0;row<N;row++){
			for(col=row-1;col<=row+1;col++){
				/* first row */
				if(row == 0){
					new_value = 1;
					if(col > row){
						new_value = -1;
					}
				}
				
				/* last row */
				if(row == N-1){
					new_value = 1;
					if(col < row){
						new_value = -1;
					}
				}

				/* ordinary row */
				if(row > 0 && row < N-1){
					new_value = 2;
					if(col > row || col < row){
						new_value = -1;
					}
				}

				/* set value */
				if(row >= 0 && row <= N-1 && col >=0 && col <= N-1){
					ierr = MatSetValue(A_petsc,row,col,new_value,INSERT_VALUES); CHKERRQ(ierr);
				}
			}
		}
		
		/* assemble matrix */
		ierr = MatAssemblyBegin(A_petsc,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
		ierr = MatAssemblyEnd(A_petsc,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
		
		/* if the matrix is small, then print it */
		if(N <= 15){
			ierr = MatView(A_petsc, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);	
		}
		
		/* init Petsc Vector */
		std::cout << " - init vector" << std::endl;
		Vec x_petsc, Ax_petsc;

		/* we can create vectors from assembled matrix */
		ierr = MatGetVecs(A_petsc,&x_petsc,&Ax_petsc); CHKERRQ(ierr);

		/* fill vector using Petsc - this could be performed better */
		PetscReal value;
		for (k=0; k<N; k++) {
			value = (PetscReal)get_some_value(k);
			ierr = VecSetValues(x_petsc,1,&k,&value,INSERT_VALUES); CHKERRQ(ierr);
		}
		
		ierr = VecAssemblyBegin(x_petsc); CHKERRQ(ierr);
		ierr = VecAssemblyEnd(x_petsc); CHKERRQ(ierr);

		std::cout << " - time init & fill vector matrix: " << getUnixTime() - t_start << "s" << std::endl;

		/* if the vector is small, then print it */
		if(N <= 15){
			ierr = VecView(x_petsc,	PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);	
		}
		
	#endif

	/* fill vector with some values */
	#if TEST_PETSCVECTOR
		std::cout << std::endl << "PETSCVECTOR:" << std::endl;

		t_start = getUnixTime();

		/* init PetscVector */
		std::cout << " - init vector" << std::endl;
		PetscVector x_petscvector(N);

		/* fill vector */
		std::cout << " - fill vector" << std::endl;

		for(k=0;k<N;k++){
			x_petscvector(k) = get_some_value(k);
		}	

		std::cout << " - time init & fill vector,: " << getUnixTime() - t_start << "s" << std::endl;

		/* if the vector is small, then print it */
		if(N <= 15) std::cout << "  " << x_petscvector << std::endl;	

		/* prepare result vector */
		PetscVector Ax_petscvector(N);
		
	#endif

	#if TEST_GENERALMATRIX_NONFREE
		std::cout << std::endl << "GENERALMATRIX_NONFREE:" << std::endl;

		t_start = getUnixTime();

		/* init PetscVector */
		std::cout << " - init vector" << std::endl;
		PetscVector x_gn(N);

		/* fill vector */
		std::cout << " - fill vector" << std::endl;

		for(k=0;k<N;k++){
			x_gn(k) = get_some_value(k);
		}	

		/* if the vector is small, then print it */
		if(N <= 15) std::cout << "  " << x_gn << std::endl;	


		/* initialize matrix */
		std::cout << " - init matrix from vector" << std::endl;
		LaplaceMatrix_petsc A_gn(x_gn);
		

		/* if the matrix is small, then print it */
		if(N <= 15) std::cout << "  " << A_gn << std::endl;	
		

		std::cout << " - time init & fill vector, matrix: " << getUnixTime() - t_start << "s" << std::endl;


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
		double norm_petsc;
	#endif
	#if TEST_PETSCVECTOR
		double t_petscvector = 0.0;
	#endif

	
	/* I want to see the problems with setting the vector values immediately in the norm */
	/* if I forget to set a component of Ax, then the norm will be huge */
	double default_value = std::numeric_limits<double>::max(); 

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

		#if TEST_PETSC
			ierr = VecSet(Ax_petsc, default_value); CHKERRQ(ierr); /* clean previous results */ 

			t_start = getUnixTime();
			ierr = MatMult(A_petsc, x_petsc, Ax_petsc); CHKERRQ(ierr);
			
			t = getUnixTime() - t_start;

			/* compute norm */
			ierr = VecNorm(Ax_petsc,NORM_2,&norm_petsc); CHKERRQ(ierr);

			std::cout << " petsc: " << t << "s, norm(Ax) = " << norm_petsc << std::endl;
			t_petsc += t;

			/* if the dimension is small, then show also the content */
			if(N <= 15){
				ierr = VecView(Ax_petsc, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);	
			}

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

		/* free space */
		ierr = MatDestroy(&A_petsc); CHKERRQ(ierr);
		ierr = VecDestroy(&x_petsc); CHKERRQ(ierr);
		ierr = VecDestroy(&Ax_petsc); CHKERRQ(ierr);

	#endif
	#if TEST_PETSCVECTOR
		std::cout << "petscvector:   " << t_petscvector/(double)M << std::endl;
	#endif


	PETSC_INITIALIZED = false;
	PetscFinalize();
	
}
