/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_VECTOR_UNARY_OPERATORS_H
#define MINLIN_VECTOR_UNARY_OPERATORS_H

#include "vector.h"

namespace minlin {
/*
template<class Expression>
Vector<typename Expression::template cos_v<Expression>::type>
cos(const Vector<Expression>& vec)
{
    return make_vector(cos(vec.expression()), vec.orientation());
}
*/

#define LIN_DEFINE_VECTOR_UNARY_OPERATOR(op) \
template<class Expression> \
Vector<typename Expression::template op##_v<Expression>::type> \
op(const Vector<Expression>& vec) \
{ \
    return make_vector(op(vec.expression()), vec.orientation()); \
}

LIN_DEFINE_VECTOR_UNARY_OPERATOR(abs)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(acos)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(asin)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(atan)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(ceil)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(cos)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(cosh)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(exp)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(floor)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(log)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(log10)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(sin)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(sinh)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(sqrt)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(tan)
LIN_DEFINE_VECTOR_UNARY_OPERATOR(tanh)


// Unary plus and minus
// ********************

template<class Expression>
Vector<typename Expression::template plus_v<Expression>::type>
operator+(const Vector<Expression>& vec)
{
    return make_vector(+vec.expression(), vec.orientation());
}

template<class Expression>
Vector<typename Expression::template minus_v<Expression>::type>
operator-(const Vector<Expression>& vec)
{
    return make_vector(-vec.expression(), vec.orientation());
}

} // end namespace minlin

#endif
