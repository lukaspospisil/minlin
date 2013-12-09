/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_REDUCTION_OPERATORS_H
#define THREX_REDUCTION_OPERATORS_H

#include "detail/expression_types.h"

#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>

namespace minlin {

namespace threx {

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
typename Expression::value_type>::type
min(const Expression& expression)
{
    return *thrust::min_element(expression.begin(), expression.end());
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
typename Expression::value_type>::type
max(const Expression& expression)
{
    return *thrust::max_element(expression.begin(), expression.end());
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
typename Expression::value_type>::type
sum(const Expression& expression)
{
    return thrust::reduce(expression.begin(), expression.end());
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
typename Expression::value_type>::type
prod(const Expression& expression)
{
    typedef typename Expression::value_type value_type;
    return thrust::reduce(
        expression.begin(), expression.end(),
        value_type(1), thrust::multiplies<value_type>()
    );
}

template<class LeftExpression, class RightExpression>
typename detail::expressions_enabler<LeftExpression::is_expression, RightExpression::is_expression,
typename LeftExpression::value_type>::type
dot(const LeftExpression& left, const RightExpression& right)
{
	typedef typename LeftExpression::value_type value_type;
    return thrust::inner_product(left.begin(), left.end(), right.begin(), value_type());
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
typename Expression::value_type>::type
norm(const Expression& expression)
{
	using std::sqrt;
    return sqrt(dot(expression, expression));
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
typename Expression::value_type>::type
norm(const Expression& expression, double p)
{
    typedef typename Expression::value_type value_type;
    if (p == std::numeric_limits<double>::infinity()) {
        // max-norm
        return thrust::transform_reduce(expression.begin(), expression.end(),
                                        detail::absVFunctor<value_type>(),
                                        value_type(),
                                        thrust::maximum<value_type>());

    } else if (p == -std::numeric_limits<double>::infinity()) {
        // min-"norm"
        return thrust::transform_reduce(expression.begin(), expression.end(),
                                        detail::absVFunctor<value_type>(),
                                        std::numeric_limits<double>::infinity(),
                                        thrust::minimum<value_type>());

    } else if (p == 1.0) {
        // one-norm
        return thrust::transform_reduce(expression.begin(), expression.end(),
                                        detail::absVFunctor<value_type>(),
                                        value_type(),
                                        thrust::plus<value_type>());
    } else if (p == 2.0) {
        // two-norm
        return norm(expression);
    } else {
        // p-norm
        using std::pow;
        return std::pow(thrust::transform_reduce(expression.begin(), expression.end(),
                                        detail::powVFunctor<value_type>(p),
                                        value_type(),
                                        thrust::plus<value_type>()), 1.0/p);
    }
}

} // end namespace threx

} // end namespace minlin

#endif
