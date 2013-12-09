/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_RELATIONAL_OPERATORS_H
#define THREX_RELATIONAL_OPERATORS_H

#include "detail/expression_types.h"
#include "detail/relational_expressions.h"

namespace minlin {

namespace threx {

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::SequalToV<const Expression> >::type
operator==(typename Expression::value_type value, const Expression& expression)
{
    return detail::SequalToV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::SnotEqualToV<const Expression> >::type
operator!=(typename Expression::value_type value, const Expression& expression)
{
    return detail::SnotEqualToV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::SlessThanV<const Expression> >::type
operator<(typename Expression::value_type value, const Expression& expression)
{
    return detail::SlessThanV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::SlessThanOrEqualToV<const Expression> >::type
operator<=(typename Expression::value_type value, const Expression& expression)
{
    return detail::SlessThanOrEqualToV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::SgreaterThanV<const Expression> >::type
operator>(typename Expression::value_type value, const Expression& expression)
{
    return detail::SgreaterThanV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::SgreaterThanOrEqualToV<const Expression> >::type
operator>=(typename Expression::value_type value, const Expression& expression)
{
    return detail::SgreaterThanOrEqualToV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VequalToS<const Expression> >::type
operator==(const Expression& expression, typename Expression::value_type value)
{
    return detail::VequalToS<const Expression>(expression, value);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VnotEqualToS<const Expression> >::type
operator!=(const Expression& expression, typename Expression::value_type value)
{
    return detail::VnotEqualToS<const Expression>(expression, value);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VlessThanS<const Expression> >::type
operator<(const Expression& expression, typename Expression::value_type value)
{
    return detail::VlessThanS<const Expression>(expression, value);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VlessThanOrEqualToS<const Expression> >::type
operator<=(const Expression& expression, typename Expression::value_type value)
{
    return detail::VlessThanOrEqualToS<const Expression>(expression, value);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VgreaterThanS<const Expression> >::type
operator>(const Expression& expression, typename Expression::value_type value)
{
    return detail::VgreaterThanS<const Expression>(expression, value);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VgreaterThanOrEqualToS<const Expression> >::type
operator>=(const Expression& expression, typename Expression::value_type value)
{
    return detail::VgreaterThanOrEqualToS<const Expression>(expression, value);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VequalToV<const LeftExpression, const RightExpression> >::type
operator==(const LeftExpression& left, const RightExpression& right)
{
    return detail::VequalToV<const LeftExpression, const RightExpression>(left, right);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VnotEqualToV<const LeftExpression, const RightExpression> >::type
operator!=(const LeftExpression& left, const RightExpression& right)
{
    return detail::VnotEqualToV<const LeftExpression, const RightExpression>(left, right);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VlessThanV<const LeftExpression, const RightExpression> >::type
operator<(const LeftExpression& left, const RightExpression& right)
{
    return detail::VlessThanV<const LeftExpression, const RightExpression>(left, right);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VlessThanOrEqualToV<const LeftExpression, const RightExpression> >::type
operator<=(const LeftExpression& left, const RightExpression& right)
{
    return detail::VlessThanOrEqualToV<const LeftExpression, const RightExpression>(left, right);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VgreaterThanV<const LeftExpression, const RightExpression> >::type
operator>(const LeftExpression& left, const RightExpression& right)
{
    return detail::VgreaterThanV<const LeftExpression, const RightExpression>(left, right);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VgreaterThanOrEqualToV<const LeftExpression, const RightExpression> >::type
operator>=(const LeftExpression& left, const RightExpression& right)
{
    return detail::VgreaterThanOrEqualToV<const LeftExpression, const RightExpression>(left, right);
}

} // end namespace threx

} // end namespace minlin

#endif
