/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_UNARY_EXPRESSIONS_H
#define THREX_DETAIL_UNARY_EXPRESSIONS_H

#include "expression_types.h"
#include "unary_functors.h"

#include <thrust/iterator/transform_iterator.h>

#include <cmath>

namespace minlin {

namespace threx {

namespace detail {
/*
template<class Expression>
struct cosV : public ExpressionType {
    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;
    typedef ::thrust::transform_iterator<cosVFunctor<value_type>, typename expression_type::const_iterator, value_type> const_iterator;
    typedef const_iterator iterator;

    explicit cosV(const expression_type& expression) : expression(expression) {}

    const_iterator begin() const {
        return const_iterator(expression.begin(), cosVFunctor<value_type>());
    }

    const_iterator end() const {
        return const_iterator(expression.end(), cosVFunctor<value_type>());
    }

    difference_type rows() const {
        return expression.rows();
    }

    difference_type cols() const {
        return expression.cols();
    }

    difference_type size() const {
        return expression.size();
    }

    value_type operator[](difference_type i) const
    {
        using std::cos;
        return cos(expression[i]);
    }

	const expression_type& expression;
};
*/

#define THREX_DEFINE_UNARY_EXPRESSION(op) \
template<class Expression> \
struct op##V : public ExpressionType { \
    typedef Expression expression_type; \
 \
    typedef typename expression_type::value_type value_type; \
    typedef typename expression_type::difference_type difference_type; \
    struct reference{}; \
    typedef reference const_reference; \
    struct pointer{}; \
    typedef pointer const_pointer; \
    typedef ::thrust::transform_iterator<op##VFunctor<value_type>, typename expression_type::const_iterator, value_type> const_iterator; \
    typedef const_iterator iterator; \
 \
    explicit op##V(const expression_type& expression) : expression(expression) {} \
 \
    const_iterator begin() const { \
        return const_iterator(expression.begin(), op##VFunctor<value_type>()); \
    } \
 \
    const_iterator end() const { \
        return const_iterator(expression.end(), op##VFunctor<value_type>()); \
    } \
 \
    difference_type rows() const { \
        return expression.rows(); \
    } \
\
    difference_type cols() const { \
        return expression.cols(); \
    } \
\
    difference_type size() const { \
        return expression.size(); \
    } \
 \
    value_type operator[](difference_type i) const \
    { \
        using std::op; \
        return op(expression[i]); \
    } \
 \
	const expression_type& expression; \
};

THREX_DEFINE_UNARY_EXPRESSION(abs)
THREX_DEFINE_UNARY_EXPRESSION(acos)
THREX_DEFINE_UNARY_EXPRESSION(asin)
THREX_DEFINE_UNARY_EXPRESSION(atan)
THREX_DEFINE_UNARY_EXPRESSION(ceil)
THREX_DEFINE_UNARY_EXPRESSION(cos)
THREX_DEFINE_UNARY_EXPRESSION(cosh)
THREX_DEFINE_UNARY_EXPRESSION(exp)
THREX_DEFINE_UNARY_EXPRESSION(floor)
THREX_DEFINE_UNARY_EXPRESSION(log)
THREX_DEFINE_UNARY_EXPRESSION(log10)
THREX_DEFINE_UNARY_EXPRESSION(sin)
THREX_DEFINE_UNARY_EXPRESSION(sinh)
THREX_DEFINE_UNARY_EXPRESSION(sqrt)
THREX_DEFINE_UNARY_EXPRESSION(tan)
THREX_DEFINE_UNARY_EXPRESSION(tanh)

template<class Expression>
struct plusV : public ExpressionType {
    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;
    typedef ::thrust::transform_iterator<plusVFunctor<value_type>, typename expression_type::const_iterator, value_type> const_iterator;
    typedef const_iterator iterator;

    explicit plusV(const expression_type& expression) : expression(expression) {}

    const_iterator begin() const {
        return const_iterator(expression.begin(), plusVFunctor<value_type>());
    }

    const_iterator end() const {
        return const_iterator(expression.end(), plusVFunctor<value_type>());
    }

    difference_type rows() const {
        return expression.rows();
    }

    difference_type cols() const {
        return expression.cols();
    }

    difference_type size() const {
        return expression.size();
    }

    value_type operator[](difference_type i) const
    {
        return +expression[i];
    }

	const expression_type& expression;
};

template<class Expression>
struct minusV : public ExpressionType {
    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;
    typedef ::thrust::transform_iterator<minusVFunctor<value_type>, typename expression_type::const_iterator, value_type> const_iterator;
    typedef const_iterator iterator;

    explicit minusV(const expression_type& expression) : expression(expression) {}

    const_iterator begin() const {
        return const_iterator(expression.begin(), minusVFunctor<value_type>());
    }

    const_iterator end() const {
        return const_iterator(expression.end(), minusVFunctor<value_type>());
    }

    difference_type rows() const {
        return expression.rows();
    }

    difference_type cols() const {
        return expression.cols();
    }

    difference_type size() const {
        return expression.size();
    }

    value_type operator[](difference_type i) const
    {
        return -expression[i];
    }

	const expression_type& expression;
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
