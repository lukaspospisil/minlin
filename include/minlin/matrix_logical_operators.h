/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_MATRIX_LOGICAL_OPERATORS_H
#define MINLIN_MATRIX_LOGICAL_OPERATORS_H

#include "matrix.h"

#include <cassert>

namespace minlin {

template<class Expression>
Matrix<typename Expression::template s_or_v<Expression>::type>
operator||(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(s || mat.expression(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template s_and_v<Expression>::type>
operator&&(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(s && mat.expression(), mat.rows(), mat.cols(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template v_or_s<Expression>::type>
operator||(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(mat.expression() || s, mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template v_and_s<Expression>::type>
operator&&(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(mat.expression() && s, mat.rows(), mat.cols());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_or_v<LeftExpression, RightExpression>::type>
operator||(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(left.expression() || right.expression(), left.rows(), right.rows());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_and_v<LeftExpression, RightExpression>::type>
operator&&(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(left.expression() && right.expression(), left.rows(), right.rows());
}

template<class Expression>
Matrix<typename Expression::template not_v<Expression>::type>
operator!(const Matrix<Expression>& mat)
{
    return make_matrix(!mat.expression(), mat.rows(), mat.cols());
}

} // end namespace minlin

#endif
