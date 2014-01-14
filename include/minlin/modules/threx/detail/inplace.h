/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_INPLACE_H
#define THREX_DETAIL_INPLACE_H

// boost helpers
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/has_xxx.hpp>
#include <boost/type_traits/is_arithmetic.hpp>

#include "expression_types.h"
#include "assignment.h"

#include <iterator>

namespace minlin {

namespace threx {

namespace detail {


BOOST_MPL_HAS_XXX_TRAIT_DEF(is_specialized);

// meta-functions for specialising the assignment operations
// only the first two template parameters, E and T, should be passed
// the parameter E is the Expression type being passed
// the parameter T is the type that will be used if E satisfies the meta-function
template < typename Expression, typename T=void,
           // disable if this is a
           typename SpecializedDisabler=typename boost::disable_if<has_is_specialized<Expression> >::type,
           typename ExpressionEnabler  =typename detail::expression_enabler<Expression::is_expression,void>::type >
struct is_thrust_expression {
    static const bool value = true;
    typedef T type;
};

// specialized expressions, such as matrix-vector multiplication aren't implemented using thrust iterators
// instead, a specialized implementation must be provided
// in the case of matrix-vector multiplication, a call to BLAS gemv routine would be used
template < typename Expression, typename T=void,
           typename SpecializedEnabler=typename boost::enable_if<has_is_specialized<Expression> >::type,
           typename ExpressionEnabler=typename detail::expression_enabler<Expression::is_expression,void>::type >
struct is_specialized_expression
{
    static const bool value = true;
    typedef T type;
};

// Base class provides in-place assignment and compound assignment
template<class Derived, class ValueType>
struct InPlaceOps : public ExpressionType {

    enum {is_inplace = true};

    typedef ValueType value_type;

    // In-place assignment
    template<typename Expression>
    typename is_thrust_expression<Expression>::type
    operator=(const Expression& expression)
    {
        assign_in_place(static_cast<Derived&>(*this), expression);
    }

    template<typename Expression>
    typename is_specialized_expression<Expression>::type
    operator=(const Expression& expression)
    {
        assign_in_place_specialized(static_cast<Derived&>(*this), expression);
    }

    // In-place assignment  
    void operator=(value_type value)
    {
        assign_value_in_place(static_cast<Derived&>(*this), value);
    }

    // In-place compound assignment
    // ****************************
    template<typename Expression>
    typename detail::expression_enabler<Expression::is_expression, void>::type
    operator+=(const Expression& expression)
    {
        plus_assign_in_place(static_cast<Derived&>(*this), expression);
    }

    template<typename Expression>
    typename detail::expression_enabler<Expression::is_expression, void>::type
    operator-=(const Expression& expression)
    {
        minus_assign_in_place(static_cast<Derived&>(*this), expression);
    }

    template<typename Expression>
    typename detail::expression_enabler<Expression::is_expression, void>::type
    operator*=(const Expression& expression)
    {
        times_assign_in_place(static_cast<Derived&>(*this), expression);
    }

    template<typename Expression>
    typename detail::expression_enabler<Expression::is_expression, void>::type
    operator/=(const Expression& expression)
    {
        divide_assign_in_place(static_cast<Derived&>(*this), expression);
    }

    void operator+=(value_type value)
    {
        plus_assign_value_in_place(static_cast<Derived&>(*this), value);
    }

    void operator-=(value_type value)
    {
        minus_assign_value_in_place(static_cast<Derived&>(*this), value);
    }

    void operator*=(value_type value)
    {
        times_assign_value_in_place(static_cast<Derived&>(*this), value);
    }

    void operator/=(value_type value)
    {
        divide_assign_value_in_place(static_cast<Derived&>(*this), value);
    }

};

template<class Expression>
struct InPlace : public InPlaceOps<InPlace<Expression>, typename Expression::value_type> {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    typedef typename expression_type::reference reference;
    typedef typename expression_type::const_reference const_reference;
    typedef typename expression_type::pointer pointer;
    typedef typename expression_type::const_pointer const_pointer;
    typedef typename expression_type::iterator iterator;
    typedef typename expression_type::const_iterator const_iterator;

    typedef InPlaceOps<InPlace, value_type> base;

    explicit InPlace(expression_type& expression) : expression(expression) {}

    // Bring in the base assignment operator templates
    using base::operator=;

    // The compiler won't synthesise an assignment operator, since we 
    // have a reference member.  We want something different anyway.
    void operator=(const InPlace& other)
    {
        assign_in_place(*this, expression);
    }

    reference operator[](difference_type i)
    {
        return expression[i];
    }

    value_type operator[](difference_type i) const
    {
        return expression[i];
    }

    iterator begin() {
        return expression.begin();
    }

    const_iterator begin() const {
        return expression.begin();
    }

    iterator end() {
        return expression.end();
    }

    const_iterator end() const {
        return expression.end();
    }

    difference_type size() const {
        return expression.size();
    }

    expression_type& expression;
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
