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
		// TODO: timers & other general funny stuff 

	public:

		virtual void print(std::ostream &output) const {}; /* print the info about matrix / print matrix */
		virtual void matmult(VectorType &y, const VectorType &x) const {}; /* y = A*x */

		template<class VectorType2>
		friend std::ostream &operator<<(std::ostream &output, const GeneralMatrix<VectorType2> &matrix); /* cannot be virtual, therefore it call virtual print() */

};

/* print general matrix, call virtual print() */
template<class VectorType>
std::ostream &operator<<(std::ostream &output, const GeneralMatrix<VectorType> &matrix){
	if(DEBUG_MODE >= 100) std::cout << "(GeneralMatrixRHS)OPERATOR: <<" << std::endl;
	matrix.print(output);
	return output;
}

/* operator A*x (creates RHS to be proceeded into overloaded operator Vector = RHS */
template<class VectorType>
GeneralMatrixRHS<VectorType> operator*(const GeneralMatrix<VectorType> &matrix, const VectorType &x){
	if(DEBUG_MODE >= 100) std::cout << "(GeneralMatrixRHS)OPERATOR: *" << std::endl;
	return GeneralMatrixRHS<VectorType>(&matrix,&x);	
}


/* right hand-side vector of y=Ax - to be provided into Vector = RHS */
template<class VectorType>
class GeneralMatrixRHS{
	private:
		const GeneralMatrix<VectorType> *matrix; /* pointer to general matrix */
		const VectorType *x; /* pointer to vector */
	public:
		/* constructor: create RHS from given pointers to matrix & vector */
		GeneralMatrixRHS(const GeneralMatrix<VectorType> *newmatrix, const VectorType *newx){
			matrix = newmatrix;
			x = newx;
		}	

		void matmult(VectorType &y){ /* call multiplication function from matrix class to perform y = A*x */
			if(DEBUG_MODE >= 100) std::cout << "(GeneralMatrixRHS)FUNCTION: matmult" << std::endl;
			
			(*matrix).matmult(y, *x); 
		}	

};




} /* end of namespace */

} /* end of MinLin namespace */

#endif
