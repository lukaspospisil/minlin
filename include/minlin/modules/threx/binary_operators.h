/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_BINARY_OPERATORS_H
#define THREX_BINARY_OPERATORS_H

#include "detail/expression_types.h"
#include "detail/binary_expressions.h"

namespace minlin {

namespace threx {

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::SplusV<const Expression> >::type
operator+(typename Expression::value_type value, const Expression& expression)
{
    return detail::SplusV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::SminusV<const Expression> >::type
operator-(typename Expression::value_type value, const Expression& expression)
{
    return detail::SminusV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::StimesV<const Expression> >::type
operator*(typename Expression::value_type value, const Expression& expression)
{
    return detail::StimesV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::SdivideV<const Expression> >::type
div(typename Expression::value_type value, const Expression& expression)
{
    return detail::SdivideV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::SpowerV<const Expression> >::type
pow(typename Expression::value_type value, const Expression& expression)
{
    return detail::SpowerV<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::Satan2V<const Expression> >::type
atan2(typename Expression::value_type value, const Expression& expression)
{
    return detail::Satan2V<const Expression>(value, expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VplusS<const Expression> >::type
operator+(const Expression& expression, typename Expression::value_type value)
{
    return detail::VplusS<const Expression>(expression, value);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VminusS<const Expression> >::type
operator-(const Expression& expression, typename Expression::value_type value)
{
    return detail::VminusS<const Expression>(expression, value);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VtimesS<const Expression> >::type
operator*(const Expression& expression, typename Expression::value_type value)
{
    return detail::VtimesS<const Expression>(expression, value);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VdivideS<const Expression> >::type
operator/(const Expression& expression, typename Expression::value_type value)
{
    return detail::VdivideS<const Expression>(expression, value);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VpowerS<const Expression> >::type
pow(const Expression& expression, typename Expression::value_type value)
{
    return detail::VpowerS<const Expression>(expression, value);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::Vatan2S<const Expression> >::type
atan2(const Expression& expression, typename Expression::value_type value)
{
    return detail::Vatan2S<const Expression>(expression, value);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VplusV<const LeftExpression, const RightExpression> >::type
operator+(const LeftExpression& left, const RightExpression& right)
{
    return detail::VplusV<const LeftExpression, const RightExpression>(left, right);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VminusV<const LeftExpression, const RightExpression> >::type
operator-(const LeftExpression& left, const RightExpression& right)
{
    return detail::VminusV<const LeftExpression, const RightExpression>(left, right);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VtimesV<const LeftExpression, const RightExpression> >::type
mul(const LeftExpression& left, const RightExpression& right)
{
    return detail::VtimesV<const LeftExpression, const RightExpression>(left, right);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VdivideV<const LeftExpression, const RightExpression> >::type
div(const LeftExpression& left, const RightExpression& right)
{
    return detail::VdivideV<const LeftExpression, const RightExpression>(left, right);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::VpowerV<const LeftExpression, const RightExpression> >::type
pow(const LeftExpression& left, const RightExpression& right)
{
    return detail::VpowerV<const LeftExpression, const RightExpression>(left, right);
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
detail::Vatan2V<const LeftExpression, const RightExpression> >::type
atan2(const LeftExpression& left, const RightExpression& right)
{
    return detail::Vatan2V<const LeftExpression, const RightExpression>(left, right);
}

} // end namespace threx

} // end namespace minlin

#endif
