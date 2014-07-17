/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_MATRIX_UNARY_OPERATORS_H
#define MINLIN_MATRIX_UNARY_OPERATORS_H

#include "vector.h"
#include "matrix.h"

#include <cassert>

namespace minlin {
/*
template<class Expression>
Matrix<typename Expression::template cos_v<Expression>::type>
cos(const Matrix<Expression>& mat)
{
    return make_matrix(cos(mat.expression()), mat.rows(), mat.cols());
}
*/

#define LIN_DEFINE_MATRIX_UNARY_OPERATOR(op) \
template<class Expression> \
Matrix<typename Expression::template op##_v<Expression>::type> \
op(const Matrix<Expression>& mat) \
{ \
    return make_matrix(op(mat.expression()), mat.rows(), mat.cols()); \
}

LIN_DEFINE_MATRIX_UNARY_OPERATOR(abs)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(acos)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(asin)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(atan)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(ceil)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(cos)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(cosh)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(exp)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(floor)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(log)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(log10)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(sin)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(sinh)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(sqrt)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(tan)
LIN_DEFINE_MATRIX_UNARY_OPERATOR(tanh)

template<class Expression>
Matrix<typename Expression::template plus_v<Expression>::type>
operator+(const Matrix<Expression>& mat)
{
    return make_matrix(+mat.expression(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template minus_v<Expression>::type>
operator-(const Matrix<Expression>& mat)
{
    return make_matrix(-mat.expression(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template transpose_m<Expression>::type>
transpose(const Matrix<Expression>& mat)
{
    // swap mat.cols and mat.rows
    typedef typename Expression::template transpose_m<Expression>::type expression_type;

    return make_matrix(expression_type(mat.expression()), mat.cols(), mat.rows());
}

template<class Expression>
typename Expression::difference_type
numel(const Matrix<Expression>& mat)
{
    return mat.size();
}

template<class Expression>
typename Expression::difference_type
size(const Matrix<Expression>& mat, typename Expression::difference_type dimension)
{
    #ifdef MINLIN_DEBUG
    assert(dimension > 0);
    #endif
    if (dimension == 1) {
        return mat.rows();
    } else if (dimension == 2) {
        return mat.cols();
    } else {
        return 1;
    }
}

} // end namespace minlin

#endif
