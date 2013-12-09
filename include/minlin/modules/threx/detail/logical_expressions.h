/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_LOGICAL_EXPRESSIONS_H
#define THREX_DETAIL_LOGICAL_EXPRESSIONS_H

#include "expression_types.h"
#include "logical_functors.h"

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace minlin {

namespace threx {

namespace detail {

template<class Expression>
struct SorV : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<SorVFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    SorV(argument_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return value || expression[i];
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), SorVFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), SorVFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct SandV : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<SandVFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    SandV(argument_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return value && expression[i];
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), SandVFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), SandVFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct VorS : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VorSFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VorS(const expression_type& expression, argument_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return expression[i] || value;
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), VorSFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), VorSFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct VandS : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VandSFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VandS(const expression_type& expression, argument_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return expression[i] && value;
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), VandSFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), VandSFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class LeftExpression, class RightExpression>
struct VorV : public ExpressionType {

	typedef LeftExpression left_expression;
	typedef RightExpression right_expression;

    typedef bool value_type;
	typedef typename left_expression::value_type argument_type;
    typedef typename left_expression::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

	typedef ::thrust::tuple<typename left_expression::const_iterator, typename right_expression::const_iterator> tuple_type;
	typedef ::thrust::zip_iterator<tuple_type> zip_type;
	
    typedef ::thrust::transform_iterator<VorVFunctor<argument_type>, zip_type> const_iterator;
    typedef const_iterator iterator;

    VorV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        return left[i] || right[i];
    }

	const_iterator begin() const {
		zip_type zip_it(::thrust::make_tuple(left.begin(), right.begin()));
		return const_iterator(zip_it, VorVFunctor<argument_type>());
	}

	const_iterator end() const {
		zip_type zip_it(::thrust::make_tuple(left.end(), right.end()));
		return const_iterator(zip_it, VorVFunctor<argument_type>());
	}
	
	difference_type size() const {
		return left.size();
	}
	
	const left_expression& left;
	const right_expression& right;
	
};

template<class LeftExpression, class RightExpression>
struct VandV : public ExpressionType {

	typedef LeftExpression left_expression;
	typedef RightExpression right_expression;

    typedef bool value_type;
	typedef typename left_expression::value_type argument_type;
    typedef typename left_expression::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

	typedef ::thrust::tuple<typename left_expression::const_iterator, typename right_expression::const_iterator> tuple_type;
	typedef ::thrust::zip_iterator<tuple_type> zip_type;
	
    typedef ::thrust::transform_iterator<VandVFunctor<argument_type>, zip_type> const_iterator;
    typedef const_iterator iterator;

    VandV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        return left[i] && right[i];
    }

	const_iterator begin() const {
		zip_type zip_it(::thrust::make_tuple(left.begin(), right.begin()));
		return const_iterator(zip_it, VandVFunctor<argument_type>());
	}

	const_iterator end() const {
		zip_type zip_it(::thrust::make_tuple(left.end(), right.end()));
		return const_iterator(zip_it, VandVFunctor<argument_type>());
	}
	
	difference_type size() const {
		return left.size();
	}
	
	const left_expression& left;
	const right_expression& right;
	
};

template<class Expression>
struct notV : public ExpressionType {
    typedef Expression expression_type;

    typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;
    typedef ::thrust::transform_iterator<notVFunctor<argument_type>, typename expression_type::const_iterator, value_type> const_iterator;
    typedef const_iterator iterator;

    explicit notV(const expression_type& expression) : expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return !expression[i];
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), notVFunctor<argument_type>());
    }

    const_iterator end() const {
        return const_iterator(expression.end(), notVFunctor<argument_type>());
    }

    difference_type size() const {
        return expression.size();
    }

	const expression_type& expression;
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
