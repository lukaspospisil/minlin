/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_VECTOR_REDUCTION_OPERATORS_H
#define MINLIN_VECTOR_REDUCTION_OPERATORS_H

#include "vector.h"

#include <cassert>

namespace minlin {

template<class Expression>
typename Expression::value_type
min(const Vector<Expression>& vec)
{
    #ifdef MINLIN_DEBUG
    assert(vec.size() != 0);
    #endif
    return min(vec.expression());
}

template<class Expression>
typename Expression::value_type
max(const Vector<Expression>& vec)
{
    #ifdef MINLIN_DEBUG
    assert(vec.size() != 0);
    #endif
    return max(vec.expression());
}

template<class Expression>
typename Expression::value_type
sum(const Vector<Expression>& vec)
{
    return sum(vec.expression());
}

template<class Expression>
typename Expression::value_type
prod(const Vector<Expression>& vec)
{
    return prod(vec.expression());
}

template<class Expression>
typename Expression::value_type
norm(const Vector<Expression>& vec)
{
    return norm(vec.expression());
}

template<class Expression>
typename Expression::value_type
norm(const Vector<Expression>& vec, double p)
{
    return norm(vec.expression(), p);
}

template<class LeftExpression, class RightExpression>
typename LeftExpression::value_type
dot(const Vector<LeftExpression>& left, const Vector<RightExpression>& right)
{
    #ifdef MINLIN_DEBUG
    assert(left.rows() == right.rows() && left.cols() == right.cols());
    #endif
    return dot(left.expression(), right.expression());
}

} // end namespace minlin

#endif
