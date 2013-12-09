/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_VECTOR_LOGICAL_OPERATORS_H
#define MINLIN_VECTOR_LOGICAL_OPERATORS_H

#include "vector.h"

#include <cassert>

namespace minlin {

template<class Expression>
Vector<typename Expression::template s_or_v<Expression>::type>
operator||(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(s || vec.expression(), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template s_and_v<Expression>::type>
operator&&(typename Expression::value_type s, const Vector<Expression>& vec)
{
    return make_vector(s && vec.expression(), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_or_s<Expression>::type>
operator||(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(vec.expression() || s, vec.orientation());
}

template<class Expression>
Vector<typename Expression::template v_and_s<Expression>::type>
operator&&(const Vector<Expression>& vec, typename Expression::value_type s)
{
    return make_vector(vec.expression() && s, vec.orientation());
}

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_or_v<LeftExpression, RightExpression>::type>
operator||(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(left.expression() || right.expression(), left.orientation());
}

template<class LeftExpression, class RightExpression>
Vector<typename LeftExpression::template v_and_v<LeftExpression, RightExpression>::type>
operator&&(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return make_vector(left.expression() && right.expression(), left.orientation());
}

template<class Expression>
Vector<typename Expression::template not_v<Expression>::type>
operator!(const Vector<Expression>& vec)
{
    return make_vector(!vec.expression(), vec.orientation());
}

} // end namespace minlin

#endif
