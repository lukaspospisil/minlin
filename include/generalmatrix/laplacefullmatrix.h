#ifndef LAPLACEFULLMATRIX_H
#define	LAPLACEFULLMATRIX_H

extern int DEBUG_MODE;

#include <iostream>
#include "generalmatrix.h"


namespace minlin {

namespace threx { // TODO: maybe choose the different namespace for my own stuff

/* --------------- LAPLACE FULL (NON-FREE) MATRIX -------------- */

/* laplace matrix */ 
template<class VectorType>
class LaplaceFullMatrix: public GeneralMatrix<VectorType> {
	private:
		/* Petsc stuff */
		Mat A_petsc;

		/* MINLIN stuff */
		HostMatrix<double> A_minlinhost;
		DeviceMatrix<double> A_minlindevice;
		
	
	public:
		LaplaceFullMatrix(const VectorType &x); /* constructor from vector */
		~LaplaceFullMatrix(); /* destructor - destroy inner matrix */

		void print(std::ostream &output) const; /* print matrix */
		void matmult(VectorType &y, const VectorType &x) const; /* y = A*x */

};



/* -------------------------------- PETSC VECTOR -------------------------*/

/* Petsc: constructor from given right PetscVector */
template<>
LaplaceFullMatrix<PetscVector>::LaplaceFullMatrix(const PetscVector &x){
	/* init Petsc Vector */
	if(DEBUG_MODE >= 100) std::cout << "(LAPLACEFULL)CONSTRUCTOR: from PetscVector" << std::endl;

	int N, n;

	/* get informations from given vector */
	N = x.size();
	n = x.local_size();

	TRY( MatCreate(PETSC_COMM_WORLD,&A_petsc) );
	TRY( MatSetSizes(A_petsc,n,n,N,N) );
	TRY( MatSetFromOptions(A_petsc) ); 
//	TRY( MatSetType(A,MATMPIAIJ) ); 
	TRY( MatMPIAIJSetPreallocation(A_petsc,5,NULL,5,NULL) ); 
	TRY( MatSeqAIJSetPreallocation(A_petsc,5,NULL) );

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

			// TODO: only for testing string problem - regularization - remove this hotfix 
			if(true){
				if((row == 0 && col == 1) || (row == 1 && col == 0) || (row == N-2 && col == N-1) || (row == N-1 && col == N-2)){
					new_value = 0;
				}
			}

			/* set value */
			if(row >= 0 && row <= N-1 && col >=0 && col <= N-1){
				TRY( MatSetValue(A_petsc,row,col,new_value,INSERT_VALUES) );
			}
		}
	}
		
	/* assemble matrix */
	TRY( MatAssemblyBegin(A_petsc,MAT_FINAL_ASSEMBLY) );
	TRY( MatAssemblyEnd(A_petsc,MAT_FINAL_ASSEMBLY) );
	
}

/* Petsc: destructor - destroy the matrix */
template<>
LaplaceFullMatrix<PetscVector>::~LaplaceFullMatrix(){
	/* init Petsc Vector */
	if(DEBUG_MODE >= 100) std::cout << "(LAPLACEFULL)DECONSTRUCTOR" << std::endl;

	if(PETSC_INITIALIZED){ /* maybe Petsc was already finalized and there is nothing to destroy */
		TRY( MatDestroy(&A_petsc) );
	}
}

/* print matrix */
template<>
void LaplaceFullMatrix<PetscVector>::print(std::ostream &output) const		
{
	if(DEBUG_MODE >= 100) std::cout << "(LAPLACEFULL)OPERATOR: << print" << std::endl;

	output << "Laplace matrix (sorry, 'only' MatView from Petsc follows):" << std::endl;
	output << "----------------------------------------------------------" << std::endl;
	
	TRY( MatView(A_petsc, PETSC_VIEWER_STDOUT_WORLD) );

	output << "----------------------------------------------------------" << std::endl;
}

/* Petsc: matrix-vector multiplication */
template<>
void LaplaceFullMatrix<PetscVector>::matmult(PetscVector &y, const PetscVector &x) const { 
	if(DEBUG_MODE >= 100) std::cout << "(LaplaceFullMatrix)FUNCTION: matmult" << std::endl;

	// TODO: maybe y is not initialized, who knows
	
	TRY( MatMult(A_petsc, x.get_vector(), y.get_vector()) ); // TODO: I dont want to use get_vector :( friend in PetscVector? and in MinLin?
}



/* -------------------------------- MINLIN HOST -------------------------*/

typedef HostVector<double> HostVectorD; /* template template is too dummy */

/* MinLinHost: constructor from given right HostVector<double> */
template<>
LaplaceFullMatrix<HostVectorD>::LaplaceFullMatrix(const HostVectorD &x){
	/* init Petsc Vector */
	if(DEBUG_MODE >= 100) std::cout << "(LAPLACEFULL)CONSTRUCTOR: from MinLin host" << std::endl;

	int N = x.size();

	/* get informations from given vector */
	HostMatrix<double> A_new(N,N);

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

			// TODO: only for testing string problem - regularization - remove this hotfix 
			if(true){
				if((row == 0 && col == 1) || (row == 1 && col == 0) || (row == N-2 && col == N-1) || (row == N-1 && col == N-2)){
					new_value = 0;
				}
			}

			/* set value */
			if(row >= 0 && row <= N-1 && col >=0 && col <= N-1){
				A_new(row,col) = new_value;
			}
		}
	}
		
	A_minlinhost = A_new;

}


/* MinLinHost: destructor - destroy the matrix */
template<>
LaplaceFullMatrix<HostVectorD>::~LaplaceFullMatrix(){
	/* init Petsc Vector */
	if(DEBUG_MODE >= 100) std::cout << "(LAPLACEFULL)DECONSTRUCTOR" << std::endl;

	// TODO: how to destroy minlin matrix?
}

/* print matrix */
template<>
void LaplaceFullMatrix<HostVectorD>::print(std::ostream &output) const		
{
	if(DEBUG_MODE >= 100) std::cout << "(LAPLACEFULL)OPERATOR: << print" << std::endl;
	output << A_minlinhost << std::endl;
}

/* MinLinHost: matrix-vector multiplication */
template<>
void LaplaceFullMatrix<HostVectorD>::matmult(HostVectorD &y, const HostVectorD &x) const { 
	if(DEBUG_MODE >= 100) std::cout << "(LaplaceFullMatrix)FUNCTION: matmult" << std::endl;

	y = A_minlinhost*x;	

}




} /* end of namespace */

} /* end of MinLin namespace */

#endif
