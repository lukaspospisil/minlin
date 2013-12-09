/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_VECTOR_BINARY_OPERATORS_H
#define MINLIN_VECTOR_BINARY_OPERATORS_H

#include "vector.h"

#include <cassert>

namespace minlin {

// Scalar-Vector operators
// ***********************

template<class Expression>
Vector<typename Expression::template s_plus_v<Expression>::type>
operator+(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(s + vec.expression(), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template s_minus_v<Expression>::type>
operator-(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(s - vec.expression(), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template s_times_v<Expression>::type>
operator*(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(s * vec.expression(), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template s_divide_v<Expression>::type>
div(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(div(s, vec.expression()), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template s_power_v<Expression>::type>
pow(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(pow(s, vec.expression()), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template s_atan2_v<Expression>::type>
atan2(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(atan2(s, vec.expression()), vec.orientation());
}

// Vector-Scalar operators
// ***********************

template<class Expression>
Vector<typename Expression::template v_plus_s<Expression>::type>
operator+(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(vec.expression() + s, vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_minus_s<Expression>::type>
operator-(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(vec.expression() - s, vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_times_s<Expression>::type>
operator*(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(vec.expression() * s, vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_divide_s<Expression>::type>
operator/(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(vec.expression() / s, vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_power_s<Expression>::type>
pow(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(pow(vec.expression(), s), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_atan2_s<Expression>::type>
atan2(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(atan2(vec.expression(), s), vec.orientation());
}

// Vector-Vector operators
// ***********************

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_plus_v<LeftExpression, RightExpression>::type>
operator+(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(left.expression() + right.expression(), left.orientation());
}

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_minus_v<LeftExpression, RightExpression>::type>
operator-(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(left.expression() - right.expression(), left.orientation());
}

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_times_v<LeftExpression, RightExpression>::type>
mul(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(mul(left.expression(), right.expression()), left.orientation());
}

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_divide_v<LeftExpression, RightExpression>::type>
div(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(div(left.expression(), right.expression()), left.orientation());
}

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_power_v<LeftExpression, RightExpression>::type>
pow(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(pow(left.expression(), right.expression()), left.orientation());
}

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_atan2_v<LeftExpression, RightExpression>::type>
atan2(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(atan2(left.expression(), right.expression()), left.orientation());
}

} // end namespace minlin

#endif
