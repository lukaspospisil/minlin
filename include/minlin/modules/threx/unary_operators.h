/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_UNARY_OPERATORS_H
#define THREX_UNARY_OPERATORS_H

#include "detail/expression_types.h"
#include "detail/unary_expressions.h"

#include <thrust/transform_reduce.h>

#include <cmath>
#include <limits>

namespace minlin {

namespace threx {
/*
template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::cosV<const Expression> >::type
cos(const Expression& expression)
{
    return detail::cosV<const Expression>(expression);
}
*/

#define THREX_DEFINE_UNARY_OPERATOR(op) \
template<class Expression> \
typename detail::expression_enabler<Expression::is_expression, \
detail::op##V<const Expression> >::type \
op(const Expression& expression) \
{ \
    return detail::op##V<const Expression>(expression); \
} \

THREX_DEFINE_UNARY_OPERATOR(abs)
THREX_DEFINE_UNARY_OPERATOR(acos)
THREX_DEFINE_UNARY_OPERATOR(asin)
THREX_DEFINE_UNARY_OPERATOR(atan)
THREX_DEFINE_UNARY_OPERATOR(ceil)
THREX_DEFINE_UNARY_OPERATOR(cos)
THREX_DEFINE_UNARY_OPERATOR(cosh)
THREX_DEFINE_UNARY_OPERATOR(exp)
THREX_DEFINE_UNARY_OPERATOR(floor)
THREX_DEFINE_UNARY_OPERATOR(log)
THREX_DEFINE_UNARY_OPERATOR(log10)
THREX_DEFINE_UNARY_OPERATOR(sin)
THREX_DEFINE_UNARY_OPERATOR(sinh)
THREX_DEFINE_UNARY_OPERATOR(sqrt)
THREX_DEFINE_UNARY_OPERATOR(tan)
THREX_DEFINE_UNARY_OPERATOR(tanh)

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::plusV<const Expression> >::type
operator+(const Expression& expression)
{
    return detail::plusV<const Expression>(expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::minusV<const Expression> >::type
operator-(const Expression& expression)
{
    return detail::minusV<const Expression>(expression);
}

} // end namespace threx

} // end namespace minlin

#endif
