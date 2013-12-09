/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_INPLACE_H
#define THREX_DETAIL_INPLACE_H

#include "expression_types.h"
#include "assignment.h"

#include <iterator>

namespace minlin {

namespace threx {

namespace detail {

// Base class provides in-place assignment and compound assignment
template<class Derived, class ValueType>
struct InPlaceOps : public ExpressionType {

    enum {is_inplace = true};

    typedef ValueType value_type;

    // In-place assignment
    template<typename Expression>
    typename detail::expression_enabler<Expression::is_expression, void>::type
    operator=(const Expression& expression)
    {
        assign_in_place(static_cast<Derived&>(*this), expression);
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
