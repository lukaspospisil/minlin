/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_RELATIONAL_EXPRESSIONS_H
#define THREX_DETAIL_RELATIONAL_EXPRESSIONS_H

#include "expression_types.h"
#include "relational_functors.h"

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace minlin {

namespace threx {

namespace detail {

template<class Expression>
struct SequalToV : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<SequalToVFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    SequalToV(argument_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return value == expression[i];
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), SequalToVFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), SequalToVFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct SnotEqualToV : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<SnotEqualToVFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    SnotEqualToV(argument_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return value != expression[i];
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), SnotEqualToVFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), SnotEqualToVFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct SlessThanV : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<SlessThanVFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    SlessThanV(argument_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return value < expression[i];
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), SlessThanVFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), SlessThanVFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct SlessThanOrEqualToV : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<SlessThanOrEqualToVFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    SlessThanOrEqualToV(argument_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return value <= expression[i];
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), SlessThanOrEqualToVFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), SlessThanOrEqualToVFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct SgreaterThanV : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<SgreaterThanVFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    SgreaterThanV(argument_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return value > expression[i];
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), SgreaterThanVFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), SgreaterThanVFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct SgreaterThanOrEqualToV : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<SgreaterThanOrEqualToVFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    SgreaterThanOrEqualToV(argument_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return value >= expression[i];
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), SgreaterThanOrEqualToVFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), SgreaterThanOrEqualToVFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct VequalToS : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VequalToSFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VequalToS(const expression_type& expression, argument_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return expression[i] == value;
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), VequalToSFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), VequalToSFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct VnotEqualToS : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VnotEqualToSFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VnotEqualToS(const expression_type& expression, argument_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return expression[i] != value;
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), VnotEqualToSFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), VnotEqualToSFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct VlessThanS : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VlessThanSFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VlessThanS(const expression_type& expression, argument_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return expression[i] < value;
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), VlessThanSFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), VlessThanSFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct VlessThanOrEqualToS : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VlessThanOrEqualToSFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VlessThanOrEqualToS(const expression_type& expression, argument_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return expression[i] <= value;
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), VlessThanOrEqualToSFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), VlessThanOrEqualToSFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct VgreaterThanS : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VgreaterThanSFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VgreaterThanS(const expression_type& expression, argument_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return expression[i] > value;
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), VgreaterThanSFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), VgreaterThanSFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};

template<class Expression>
struct VgreaterThanOrEqualToS : public ExpressionType {

	typedef Expression expression_type;

	typedef bool value_type;
	typedef typename expression_type::value_type argument_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VgreaterThanOrEqualToSFunctor<argument_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VgreaterThanOrEqualToS(const expression_type& expression, argument_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return expression[i] >= value;
    }

	const_iterator begin() const {
		return const_iterator(expression.begin(), VgreaterThanOrEqualToSFunctor<argument_type>(value));
	}

	const_iterator end() const {
		return const_iterator(expression.end(), VgreaterThanOrEqualToSFunctor<argument_type>(value));
	}
	
	difference_type size() const {
		return expression.size();
	}
	
	argument_type value;
	const expression_type& expression;
	
};


template<class LeftExpression, class RightExpression>
struct VequalToV : public ExpressionType {

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
	
    typedef ::thrust::transform_iterator<VequalToVFunctor<argument_type>, zip_type> const_iterator;
    typedef const_iterator iterator;

    VequalToV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        return left[i] == right[i];
    }

	const_iterator begin() const {
		zip_type zip_it(::thrust::make_tuple(left.begin(), right.begin()));
		return const_iterator(zip_it, VequalToVFunctor<argument_type>());
	}

	const_iterator end() const {
		zip_type zip_it(::thrust::make_tuple(left.end(), right.end()));
		return const_iterator(zip_it, VequalToVFunctor<argument_type>());
	}
	
	difference_type size() const {
		return left.size();
	}
	
	const left_expression& left;
	const right_expression& right;
	
};

template<class LeftExpression, class RightExpression>
struct VnotEqualToV : public ExpressionType {

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
	
    typedef ::thrust::transform_iterator<VnotEqualToVFunctor<argument_type>, zip_type> const_iterator;
    typedef const_iterator iterator;

    VnotEqualToV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        return left[i] != right[i];
    }

	const_iterator begin() const {
		zip_type zip_it(::thrust::make_tuple(left.begin(), right.begin()));
		return const_iterator(zip_it, VnotEqualToVFunctor<argument_type>());
	}

	const_iterator end() const {
		zip_type zip_it(::thrust::make_tuple(left.end(), right.end()));
		return const_iterator(zip_it, VnotEqualToVFunctor<argument_type>());
	}
	
	difference_type size() const {
		return left.size();
	}
	
	const left_expression& left;
	const right_expression& right;
	
};

template<class LeftExpression, class RightExpression>
struct VlessThanV : public ExpressionType {

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
	
    typedef ::thrust::transform_iterator<VlessThanVFunctor<argument_type>, zip_type> const_iterator;
    typedef const_iterator iterator;

    VlessThanV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        return left[i] < right[i];
    }

	const_iterator begin() const {
		zip_type zip_it(::thrust::make_tuple(left.begin(), right.begin()));
		return const_iterator(zip_it, VlessThanVFunctor<argument_type>());
	}

	const_iterator end() const {
		zip_type zip_it(::thrust::make_tuple(left.end(), right.end()));
		return const_iterator(zip_it, VlessThanVFunctor<argument_type>());
	}
	
	difference_type size() const {
		return left.size();
	}
	
	const left_expression& left;
	const right_expression& right;
	
};

template<class LeftExpression, class RightExpression>
struct VlessThanOrEqualToV : public ExpressionType {

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
	
    typedef ::thrust::transform_iterator<VlessThanOrEqualToVFunctor<argument_type>, zip_type> const_iterator;
    typedef const_iterator iterator;

    VlessThanOrEqualToV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        return left[i] <= right[i];
    }

	const_iterator begin() const {
		zip_type zip_it(::thrust::make_tuple(left.begin(), right.begin()));
		return const_iterator(zip_it, VlessThanOrEqualToVFunctor<argument_type>());
	}

	const_iterator end() const {
		zip_type zip_it(::thrust::make_tuple(left.end(), right.end()));
		return const_iterator(zip_it, VlessThanOrEqualToVFunctor<argument_type>());
	}
	
	difference_type size() const {
		return left.size();
	}
	
	const left_expression& left;
	const right_expression& right;
	
};

template<class LeftExpression, class RightExpression>
struct VgreaterThanV : public ExpressionType {

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
	
    typedef ::thrust::transform_iterator<VgreaterThanVFunctor<argument_type>, zip_type> const_iterator;
    typedef const_iterator iterator;

    VgreaterThanV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        return left[i] > right[i];
    }

	const_iterator begin() const {
		zip_type zip_it(::thrust::make_tuple(left.begin(), right.begin()));
		return const_iterator(zip_it, VgreaterThanVFunctor<argument_type>());
	}

	const_iterator end() const {
		zip_type zip_it(::thrust::make_tuple(left.end(), right.end()));
		return const_iterator(zip_it, VgreaterThanVFunctor<argument_type>());
	}
	
	difference_type size() const {
		return left.size();
	}
	
	const left_expression& left;
	const right_expression& right;
	
};

template<class LeftExpression, class RightExpression>
struct VgreaterThanOrEqualToV : public ExpressionType {

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
	
    typedef ::thrust::transform_iterator<VgreaterThanOrEqualToVFunctor<argument_type>, zip_type> const_iterator;
    typedef const_iterator iterator;

    VgreaterThanOrEqualToV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        return left[i] >= right[i];
    }

	const_iterator begin() const {
		zip_type zip_it(::thrust::make_tuple(left.begin(), right.begin()));
		return const_iterator(zip_it, VgreaterThanOrEqualToVFunctor<argument_type>());
	}

	const_iterator end() const {
		zip_type zip_it(::thrust::make_tuple(left.end(), right.end()));
		return const_iterator(zip_it, VgreaterThanOrEqualToVFunctor<argument_type>());
	}
	
	difference_type size() const {
		return left.size();
	}
	
	const left_expression& left;
	const right_expression& right;
	
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
