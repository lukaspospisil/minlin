/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_MATRIX_IO_H
#define MINLIN_MATRIX_IO_H

#include "matrix.h"

#include <ostream>

namespace minlin {

// Output streaming
// ****************

template<class Expression>
std::ostream& operator<<(std::ostream& os, const Matrix<Expression>& A)
{
	typedef typename Matrix<Expression>::difference_type difference_type;
	os << '\n';
	for (difference_type i = 0; i < A.rows(); ++i)
	{
		os << ' ';
		for (difference_type j = 0; j < A.cols(); ++j)
		{
			os << A(i,j) << ' ';
		}
		os << '\n';
	}
    return os;
}

} // end namespace minlin

#endif
