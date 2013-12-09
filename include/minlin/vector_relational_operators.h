/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_VECTOR_RELATIONAL_OPERATORS_H
#define MINLIN_VECTOR_RELATIONAL_OPERATORS_H

#include "vector.h"

#include <cassert>

namespace minlin {

// Scalar-Vector operators
// ***********************

template<class Expression>
Vector<typename Expression::template s_equal_to_v<Expression>::type>
operator==(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(s == vec.expression(), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template s_not_equal_to_v<Expression>::type>
operator!=(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(s != vec.expression(), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template s_less_than_v<Expression>::type>
operator<(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(s < vec.expression(), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template s_less_than_or_equal_to_v<Expression>::type>
operator<=(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(s <= vec.expression(), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template s_greater_than_v<Expression>::type>
operator>(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(s > vec.expression(), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template s_greater_than_or_equal_to_v<Expression>::type>
operator>=(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(s >= vec.expression(), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_equal_to_s<Expression>::type>
operator==(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(vec.expression() == s, vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_not_equal_to_s<Expression>::type>
operator!=(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(vec.expression() != s, vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_less_than_s<Expression>::type>
operator<(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(vec.expression() < s, vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_less_than_or_equal_to_s<Expression>::type>
operator<=(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(vec.expression() <= s, vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_greater_than_s<Expression>::type>
operator>(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(vec.expression() > s, vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_greater_than_or_equal_to_s<Expression>::type>
operator>=(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(vec.expression() >= s, vec.orientation());
}

// Vector-Vector operators
// ***********************

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_equal_to_v<LeftExpression, RightExpression>::type>
operator==(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(left.expression() == right.expression(), left.orientation());
}

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_not_equal_to_v<LeftExpression, RightExpression>::type>
operator!=(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(left.expression() != right.expression(), left.orientation());
}

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_less_than_v<LeftExpression, RightExpression>::type>
operator<(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(left.expression() < right.expression(), left.orientation());
}

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_less_than_or_equal_to_v<LeftExpression, RightExpression>::type>
operator<=(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(left.expression() <= right.expression(), left.orientation());
}

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_greater_than_v<LeftExpression, RightExpression>::type>
operator>(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(left.expression() > right.expression(), left.orientation());
}

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_greater_than_or_equal_to_v<LeftExpression, RightExpression>::type>
operator>=(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(left.expression() >= right.expression(), left.orientation());
}

} // end namespace minlin

#endif
