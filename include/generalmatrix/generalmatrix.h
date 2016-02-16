#ifndef GENERALMATRIX_H
#define	GENERALMATRIX_H

extern int DEBUG_MODE;

#include <iostream>

namespace minlin {

namespace threx { // TODO: maybe choose the different namespace for my own stuff

class GeneralMatrix {
	protected:
		int test;
		
	public:
	
		friend std::ostream &operator<<(std::ostream &output, const GeneralMatrix &matrix); /* cannot be virtual, therefore it call virtual print() */

		template<class VectorType>
		friend const VectorType &operator*(const GeneralMatrix &matrix, const VectorType &x); /* call virtual matmult */

		virtual void print(std::ostream &output) const {}; /* print the info about matrix */

		template<class VectorType>
		virtual void matmult(VectorType &y, const VectorType &x) const {}; /* y = A*x */

};


/* print general matrix, call virtual print() */
std::ostream &operator<<(std::ostream &output, const GeneralMatrix &matrix){
	matrix.print(output);
	return output;
}

/* operator A*x (input is arbitrary vector type, output gives the same result) */
template<class VectorType>
const VectorType &operator*(const GeneralMatrix &matrix, const VectorType &x){
	matrix.matmult(this, x);
	return *this;	
}



/* --------------- LAPLACE MATRIX -------------- */
//TODO: move this to different file

/* laplace matrix */ 
class LaplaceMatrix_petsc: public GeneralMatrix {
	private:
		Mat A;
	
	public:
		LaplaceMatrix_petsc(const PetscVector &x);

		void print(std::ostream &output) const;

		void matmult(PetscVector &y, const PetscVector &x); /* y = A*x */
			
};


/* constructor from given right PetscVector */
LaplaceMatrix_petsc::LaplaceMatrix_petsc(const PetscVector &x){
	/* init Petsc Vector */
	if(DEBUG_MODE >= 100) std::cout << "(LAPLACE_PETSC)CONSTRUCTOR: from PetscVector" << std::endl;

	int N, n;

	/* get informations from given vector */
	N = x.size();
	n = x.local_size();

	TRY( MatCreate(PETSC_COMM_WORLD,&A) );
	TRY( MatSetSizes(A,n,n,N,N) );
	TRY( MatSetFromOptions(A) ); 
//	TRY( MatSetType(A,MATMPIAIJ) ); 
	TRY( MatMPIAIJSetPreallocation(A,5,NULL,5,NULL) ); 
	TRY( MatSeqAIJSetPreallocation(A,5,NULL) );

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
				TRY( MatSetValue(A,row,col,new_value,INSERT_VALUES) );
			}
		}
	}
		
	/* assemble matrix */
	TRY( MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY) );
	TRY( MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY) );
	
}

/* print matrix */
void LaplaceMatrix_petsc::print(std::ostream &output) const		
{
	if(DEBUG_MODE >= 100) std::cout << "(LAPLACE_PETSC)OPERATOR: << print" << std::endl;

	output << "Laplace matrix (sorry, 'only' MatView from Petsc follows):" << std::endl;
	output << "----------------------------------------------------------" << std::endl;
	
	TRY( MatView(A, PETSC_VIEWER_STDOUT_WORLD) );

	output << "----------------------------------------------------------" << std::endl;
}

/* matrix-vector multiplication */
/*const PetscVector &operator*(const LaplaceMatrix_petsc &matrix, const PetscVector &vec){
	
	
}*/



} /* end of namespace */

} /* end of MinLin namespace */

#endif
