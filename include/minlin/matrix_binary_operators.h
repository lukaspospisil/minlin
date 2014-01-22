/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_MATRIX_BINARY_OPERATORS_H
#define MINLIN_MATRIX_BINARY_OPERATORS_H

#include "matrix.h"

#include <cassert>

namespace minlin {

// Scalar-Matrix operators
// ***********************

template<class Expression>
Matrix<typename Expression::template s_plus_v<Expression>::type>
operator+(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(s + mat.expression(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template s_minus_v<Expression>::type>
operator-(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(s - mat.expression(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template s_times_v<Expression>::type>
operator*(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(s * mat.expression(), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template s_divide_v<Expression>::type>
div(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(div(s, mat.expression()), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template s_power_v<Expression>::type>
pow(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(pow(s, mat.expression()), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template s_atan2_v<Expression>::type>
atan2(typename Expression::value_type s, const Matrix<Expression>& mat)
{
    return make_matrix(atan2(s, mat.expression()), mat.rows(), mat.cols());
}

// Matrix-Scalar operators
// ***********************

template<class Expression>
Matrix<typename Expression::template v_plus_s<Expression>::type>
operator+(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(mat.expression() + s, mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template v_minus_s<Expression>::type>
operator-(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(mat.expression() - s, mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template v_times_s<Expression>::type>
operator*(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(mat.expression() * s, mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template v_divide_s<Expression>::type>
operator/(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(mat.expression() / s, mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template v_power_s<Expression>::type>
pow(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(pow(mat.expression(), s), mat.rows(), mat.cols());
}

template<class Expression>
Matrix<typename Expression::template v_atan2_s<Expression>::type>
atan2(const Matrix<Expression>& mat, typename Expression::value_type s)
{
    return make_matrix(atan2(mat.expression(), s), mat.rows(), mat.cols());
}

// Matrix-Matrix operators
// ***********************

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_plus_v<LeftExpression, RightExpression>::type>
operator+(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(left.expression() + right.expression(), left.rows(), left.cols());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_minus_v<LeftExpression, RightExpression>::type>
operator-(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(left.expression() - right.expression(), left.rows(), left.cols());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_times_v<LeftExpression, RightExpression>::type>
mul(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(mul(left.expression(), right.expression()), left.rows(), left.cols());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_divide_v<LeftExpression, RightExpression>::type>
div(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(div(left.expression(), right.expression()), left.rows(), left.cols());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_power_v<LeftExpression, RightExpression>::type>
pow(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(pow(left.expression(), right.expression()), left.rows(), left.cols());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template v_atan2_v<LeftExpression, RightExpression>::type>
atan2(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_matrix(atan2(left.expression(), right.expression()), left.rows(), left.cols());
}

template<class LeftExpression, class RightExpression>
Matrix<typename LeftExpression::template m_times_m<LeftExpression, RightExpression>::type>
operator*(const Matrix<LeftExpression>& left, const Matrix<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    std::cout << "matrix matrix multiplication" << std::endl;
    assert(left.cols() == right.rows());
    #endif
    typedef typename LeftExpression::template m_times_m<LeftExpression, RightExpression>::type expression_type;
    // need to provide interface for m, n, k to be passed to expression_type
    return make_matrix( expression_type(left.expression(), right.expression(), left.rows(), right.cols(), left.cols()),
                        left.rows(), right.cols());
}

} // end namespace minlin

#endif
