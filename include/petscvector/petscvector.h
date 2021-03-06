#ifndef PETSCVECTOR_H
#define	PETSCVECTOR_H

extern int DEBUG_MODE;
extern bool PETSC_INITIALIZED;

#include <iostream>
#include <string>
#include <list>

static PetscErrorCode ierr;
#define TRY( f) {ierr = f; do {if (PetscUnlikely(ierr)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr,PETSC_ERROR_IN_CXX,0);}} while(0);}

namespace minlin {

namespace threx { // TODO: maybe choose the different namespace for my own Petsc stuff
 
class PetscVectorWrapperComb; /* wrapper to allow manipulation with linear combinations of vectors */
class PetscVectorWrapperCombNode; /* one node of previous wrapper */
class PetscVectorWrapperSub; /* wrapper to allow subvectors */

/* class for manipulation with A*x as a RHS */
template<class VectorType>
class GeneralMatrixRHS;


/*! \class PetscVector
    \brief Class for manipulation with PetscVector.

    Here will be a detailed description.
*/
class PetscVector {
	private:
		Vec inner_vector; /* original Petsc Vector */
		
	public:

		PetscVector();
		PetscVector(int n);
		PetscVector(const PetscVector &vec1);
		PetscVector(Vec new_inner_vector);
		PetscVector(double scalar_value);
		
		~PetscVector();

		void valuesUpdate() const;
		void scale(PetscScalar alpha);

		Vec get_vector() const; // TODO: temp, direct access to inner vector should be forbidden
		int size() const;
		int local_size() const;

		double get(int index);
		void get_ownership(int *low, int *high);

		void get_array(double **arr);
		void restore_array(double **arr);

		void set(double new_value);
		void set(int index, double new_value);

		/* assignment */
		PetscVector &operator=(const PetscVector &vec2);
		PetscVector &operator=(PetscVectorWrapperCombNode combnode);	
		PetscVector &operator=(PetscVectorWrapperComb comb);	
		PetscVector &operator=(double scalar_value);	
		
		/* subvector */
		PetscVectorWrapperSub operator()(int index) const;
		PetscVectorWrapperSub operator()(int index_begin,int index_end) const;
		PetscVectorWrapperSub operator()(const IS new_subvector_is) const;

		friend std::ostream &operator<<(std::ostream &output, const PetscVector &vector);

		friend void operator*=(PetscVector &vec1, double alpha);
		friend void operator+=(PetscVector &vec1, const PetscVectorWrapperComb comb);
		friend void operator-=(PetscVector &vec1, const PetscVectorWrapperComb comb);

		friend double dot(const PetscVector &vec1, const PetscVector &vec2);
		friend double max(const PetscVector &vec1);
		friend double sum(const PetscVector &vec1);
		friend double norm(const PetscVector &vec1);
		friend const PetscVector operator/(const PetscVector &vec1, const PetscVector &vec2);

		
		/* y = A*x, where A*x is created as RHS */
		template<class VectorType>
		VectorType &operator=(GeneralMatrixRHS<VectorType> rhs){
			rhs.matmult(*this);
			return *this;
		}

		/* define operator PetscVector(all), it is a mixture of petscvector and minlin  */
		PetscVectorWrapperSub operator()(minlin::detail::all_type add_in) const;

};


/*! \class PetscVectorWrapperComb
    \brief Wrapper to allow linear combinations of vectors.

*/
class PetscVectorWrapperComb
{
	private:
		std::list<PetscVectorWrapperCombNode> comb_list; /* the list of linear combination nodes */
		
		int vector_size;
		
	public:
		PetscVectorWrapperComb();
		PetscVectorWrapperComb(const PetscVectorWrapperCombNode &comb_node);
		PetscVectorWrapperComb(const PetscVector &vec);
		PetscVectorWrapperComb(const PetscVectorWrapperSub subvector);
		PetscVectorWrapperComb(const double &scalar_value);

		~PetscVectorWrapperComb();


		int get_listsize() const;
		int get_vectorsize() const;
		Vec get_first_vector();
		void append(const PetscVectorWrapperCombNode &new_node);
		void merge(PetscVectorWrapperComb &comb);
		void get_arrays(PetscScalar *coeffs, Vec *vectors);
		
		friend std::ostream &operator<<(std::ostream &output, PetscVectorWrapperComb &wrapper); // TODO: const makes a problem with the list

		friend const PetscVectorWrapperComb operator*(double alpha, PetscVectorWrapperComb comb);
		friend const PetscVectorWrapperComb operator+(PetscVectorWrapperComb comb1, PetscVectorWrapperComb comb2);
		friend const PetscVectorWrapperComb operator-(PetscVectorWrapperComb comb1, PetscVectorWrapperComb comb2);

		
};


/*! \class PetscVectorWrapperComb
    \brief Wrapper with one node in the linear combinations.

*/
class PetscVectorWrapperCombNode
{
	private:
		Vec inner_vector; /* pointer to vector in linear combination */
		double coeff; /* coefficient in linear combination */
	
	public:
		PetscVectorWrapperCombNode(); // TODO: temp
		PetscVectorWrapperCombNode(double new_coeff, Vec new_vector);
		PetscVectorWrapperCombNode(double scalar_value);
	
		void set_vector(Vec new_vector);
		Vec get_vector() const;
		int get_size() const;
		int get_value(int index) const;
		
		void set_coeff(double new_coeff);
		void scale(double alpha);
		double get_coeff() const;

		friend std::ostream &operator<<(std::ostream &output, PetscVectorWrapperCombNode &wrapper);

};

/*! \class PetscVectorWrapperSub
    \brief Wrapper with subvectors.

*/
class PetscVectorWrapperSub
{
	private:
		IS subvector_is;
		Vec inner_vector; /* original vector */
		Vec subvector; /* subvector */

		bool free_is; /* free index set in destructor */

	public:

		PetscVectorWrapperSub(Vec inner_vector, IS subvector_is, bool new_free_is);
		~PetscVectorWrapperSub();

		void valuesUpdate() const;
		void scale(PetscScalar alpha) const;

		void set(double new_value);		
		Vec get_subvector();

		double get(int index);

		friend std::ostream &operator<<(std::ostream &output, const PetscVectorWrapperSub &wrapper);				

		PetscVectorWrapperSub &operator=(PetscVectorWrapperSub subvec); /* subvec = subvec */
		PetscVectorWrapperSub &operator=(double scalar_value);	 /* subvec = const */
		PetscVectorWrapperSub &operator=(const PetscVector &vec2); /* subvec = vec */
		PetscVectorWrapperSub &operator=(const PetscVectorWrapperComb &comb);	


//		PetscVectorWrapperSub &operator=(const PetscVectorWrapperCombNode combnode);	



		friend void operator*=(const PetscVectorWrapperSub &subvec1, double alpha); /* subvec = alpha*subvec */
		friend void operator+=(const PetscVectorWrapperSub &subvec1, const PetscVectorWrapperComb comb);
		friend void operator-=(const PetscVectorWrapperSub &subvec1, const PetscVectorWrapperComb comb);
		friend void operator/=(const PetscVectorWrapperSub &subvec1, const PetscVectorWrapperSub subvec2);

		friend bool operator==(PetscVectorWrapperSub subvec1, double alpha);
		friend bool operator==(PetscVectorWrapperSub subvec1, PetscVectorWrapperSub subvec2);

		friend bool operator>(PetscVectorWrapperSub vec1, PetscVectorWrapperSub vec2);

		friend double sum(const PetscVectorWrapperSub subvec1);
		friend double dot(const PetscVectorWrapperSub subvec1, const PetscVectorWrapperSub subvec2);


};




} /* end of namespace */

} /* end of MinLin namespace */


#include "petscvector_impl.h"
#include "wrappercomb_impl.h"
#include "wrappersub_impl.h"

#endif
