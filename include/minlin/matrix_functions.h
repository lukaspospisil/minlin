/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_MATRIX_FUNCTIONS_H
#define MINLIN_MATRIX_FUNCTIONS_H

#include "matrix.h"

namespace minlin {

template<class Expression>
bool any_of(const Matrix<Expression>& mat)
{
    return any_of(mat.expression());
}

template<class Expression>
bool all_of(const Matrix<Expression>& mat)
{
    return all_of(mat.expression());
}

} // end namespace minlin

#endif
