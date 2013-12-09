/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_LOGICAL_OPERATORS_H
#define THREX_LOGICAL_OPERATORS_H

#include "detail/expression_types.h"
#include "detail/logical_expressions.h"

namespace minlin {

namespace threx {

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::SorV<const Expression> >::type
operator||(typename Expression::value_type value, const Expression& expression)
{
    return detail::SorV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::SandV<const Expression> >::type
operator&&(typename Expression::value_type value, const Expression& expression)
{
    return detail::SandV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VorS<const Expression> >::type
operator||(const Expression& expression, typename Expression::value_type value)
{
    return detail::VorS<const Expression>(expression, value);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VandS<const Expression> >::type
operator&&(const Expression& expression, typename Expression::value_type value)
{
    return detail::VandS<const Expression>(expression, value);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VorV<const LeftExpression, const RightExpression> >::type
operator||(const LeftExpression& left, const RightExpression& right)
{
    return detail::VorV<const LeftExpression, const RightExpression>(left, right);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VandV<const LeftExpression, const RightExpression> >::type
operator&&(const LeftExpression& left, const RightExpression& right)
{
    return detail::VandV<const LeftExpression, const RightExpression>(left, right);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::notV<const Expression> >::type
operator!(const Expression& expression)
{
    return detail::notV<const Expression>(expression);
}

} // end namespace threx

} // end namespace minlin

#endif
