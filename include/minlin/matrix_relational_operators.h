/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_MATRIX_RELATIONAL_OPERATORS_H
#define MINLIN_MATRIX_RELATIONAL_OPERATORS_H

#include "matrix.h"

#include <cassert>

namespace minlin {

// Scalar-Matrix operators
// ***********************

template<class Expression>
Matrix<typename Expression::template s_equal_to_v<Expression>::type>
operator==(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(s == mat.expression(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template s_not_equal_to_v<Expression>::type>
operator!=(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(s != mat.expression(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template s_less_than_v<Expression>::type>
operator<(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(s < mat.expression(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template s_less_than_or_equal_to_v<Expression>::type>
operator<=(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(s <= mat.expression(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template s_greater_than_v<Expression>::type>
operator>(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(s > mat.expression(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template s_greater_than_or_equal_to_v<Expression>::type>
operator>=(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(s >= mat.expression(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template v_equal_to_s<Expression>::type>
operator==(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(mat.expression() == s, mat.rows());
}

template<class Expression>
Matrix<typename Expression::template v_not_equal_to_s<Expression>::type>
operator!=(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(mat.expression() != s, mat.rows());
}

template<class Expression>
Matrix<typename Expression::template v_less_than_s<Expression>::type>
operator<(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(mat.expression() < s, mat.rows());
}

template<class Expression>
Matrix<typename Expression::template v_less_than_or_equal_to_s<Expression>::type>
operator<=(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(mat.expression() <= s, mat.rows());
}

template<class Expression>
Matrix<typename Expression::template v_greater_than_s<Expression>::type>
operator>(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(mat.expression() > s, mat.rows());
}

template<class Expression>
Matrix<typename Expression::template v_greater_than_or_equal_to_s<Expression>::type>
operator>=(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(mat.expression() >= s, mat.rows());
}

// Matrix-Matrix operators
// ***********************

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_equal_to_v<LeftExpression, RightExpression>::type>
operator==(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(left.expression() == right.expression(), left.rows(), left.cols());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_not_equal_to_v<LeftExpression, RightExpression>::type>
operator!=(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(left.expression() != right.expression(), left.rows(), left.cols());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_less_than_v<LeftExpression, RightExpression>::type>
operator<(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(left.expression() < right.expression(), left.rows(), left.cols());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_less_than_or_equal_to_v<LeftExpression, RightExpression>::type>
operator<=(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(left.expression() <= right.expression(), left.rows(), left.cols());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_greater_than_v<LeftExpression, RightExpression>::type>
operator>(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(left.expression() > right.expression(), left.rows(), left.cols());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_greater_than_or_equal_to_v<LeftExpression, RightExpression>::type>
operator>=(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(left.expression() >= right.expression(), left.rows(), left.cols());
}

} // end namespace minlin

#endif
