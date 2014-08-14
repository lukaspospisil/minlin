/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_MATRIX_REDUCTION_OPERATORS_H
#define MINLIN_MATRIX_REDUCTION_OPERATORS_H

#include "matrix.h"

#include <cassert>

namespace minlin {

template<class Expression>
typename Expression::value_type
min(const Matrix<Expression>& mat)
{
    #ifdef MINLIN_DEBUG
    assert(mat.size() != 0);
    #endif
    return min(mat.expression());
}

template<class Expression>
typename Expression::value_type
max(const Matrix<Expression>& mat)
{
    #ifdef MINLIN_DEBUG
    assert(mat.size() != 0);
    #endif
    return max(mat.expression());
}

template<class Expression>
typename Expression::value_type
sum(const Matrix<Expression>& mat)
{
    return sum(mat.expression());
}

template<class Expression>
typename Expression::value_type
prod(const Matrix<Expression>& mat)
{
    return prod(mat.expression());
}

} // end namespace minlin

#endif
