#ifndef GENERALMATRIX_H
#define	GENERALMATRIX_H

extern int DEBUG_MODE;

#include <iostream>

namespace minlin {

namespace threx { // TODO: maybe choose the different namespace for my own stuff

template<class VectorType>
class GeneralMatrixRHS;

template<class VectorType>
class GeneralMatrix {
	protected:
		int test;
		
	public:
	
		template<class VectorType2>
		friend std::ostream &operator<<(std::ostream &output, const GeneralMatrix<VectorType2> &matrix); /* cannot be virtual, therefore it call virtual print() */

		virtual void print(std::ostream &output) const {}; /* print the info about matrix */

		/* we are not able to define template virtual functions :( */
		virtual void matmult(VectorType &y, const VectorType &x) const {}; /* y = A*x */

};

/* print general matrix, call virtual print() */
template<class VectorType>
std::ostream &operator<<(std::ostream &output, const GeneralMatrix<VectorType> &matrix){
	matrix.print(output);
	return output;
}

/* operator A*x (input is arbitrary vector type, output gives the same result) */
template<class VectorType>
GeneralMatrixRHS<VectorType> operator*(const GeneralMatrix<VectorType> &matrix, const VectorType &x){
	return GeneralMatrixRHS<VectorType>(&matrix,&x);	
}



/* right hand-side vector of y=Ax - to provide into LHS = RHS */
template<class VectorType>
class GeneralMatrixRHS{
	private:
		const GeneralMatrix<VectorType> *matrix;
		const VectorType *x;
	public:
		GeneralMatrixRHS(const GeneralMatrix<VectorType> *newmatrix, const VectorType *newx){
			matrix = newmatrix;
			x = newx;
		}	

		void matmult(VectorType &y){ /* y = rhs */
			if(DEBUG_MODE >= 100) std::cout << "(GeneralMatrixRHS)FUNCTION: matmult" << std::endl;
			
			(*matrix).matmult(y, *x); 
		}	

};



//TODO: move what follow to different file
/* --------------- LAPLACE MATRIX -------------- */

/* laplace matrix */ 
template<class VectorType>
class LaplaceMatrixFull: public GeneralMatrix<VectorType> {
	private:
		/* Petsc stuff */
		Mat A_petsc;
		
	
	public:
		LaplaceMatrixFull(const VectorType &x); /* constructor from vector */

		void print(std::ostream &output) const;

		void matmult(VectorType &y, const VectorType &x) const; /* y = A*x */

};


/* Petsc: constructor from given right PetscVector */
template<>
LaplaceMatrixFull<PetscVector>::LaplaceMatrixFull(const PetscVector &x){
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

/* print matrix */
template<>
void LaplaceMatrixFull<PetscVector>::print(std::ostream &output) const		
{
	if(DEBUG_MODE >= 100) std::cout << "(LAPLACEFULL)OPERATOR: << print" << std::endl;

	output << "Laplace matrix (sorry, 'only' MatView from Petsc follows):" << std::endl;
	output << "----------------------------------------------------------" << std::endl;
	
	TRY( MatView(A_petsc, PETSC_VIEWER_STDOUT_WORLD) );

	output << "----------------------------------------------------------" << std::endl;
}

/* Petsc: matrix-vector multiplication */
template<>
void LaplaceMatrixFull<PetscVector>::matmult(PetscVector &y, const PetscVector &x) const { 
	if(DEBUG_MODE >= 100) std::cout << "(LaplaceMatrixFull)FUNCTION: matmult" << std::endl;

	TRY( MatMult(A_petsc, x.get_vector(), y.get_vector()) );

}


} /* end of namespace */

} /* end of MinLin namespace */

#endif
