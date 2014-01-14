/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au 
   licensed under BSD license (see LICENSE.txt for details)

   file created by Ben Cumming

   define matrix-vector operators, such as matrix vector multiplication (gemv)
*******************************************************************************/

#pragma once

#include "matrix.h"
#include "vector.h"

#include <cassert>

namespace minlin {

// Matrix-Vector operators
// ***********************

// matrix vector multiplication
template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template m_times_v<LeftExpression, RightExpression>::type>
operator*(const Matrix<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    std::cout << "matrix-vector multiplication" << std::endl;
    assert(left.cols() == right.size()); // treat all vectors as a column vector
    #endif

    typedef typename LeftExpression::template m_times_v<LeftExpression, RightExpression>::type expression_type;
    return make_vector(expression_type(left.expression(),right.expression(),left.rows(),left.cols()), right.orientation());
}

} // end namespace minlin
