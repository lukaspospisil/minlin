/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_ASSIGNMENT_H
#define THREX_DETAIL_ASSIGNMENT_H

#include <thrust/tuple.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include "expression_types.h"

namespace minlin {

namespace threx {
    
namespace detail {

struct AssignmentFunctor
{
    template<class Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        ::thrust::get<0>(t) = ::thrust::get<1>(t);
    }
};

struct PlusAssignmentFunctor
{
    template<class Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        ::thrust::get<0>(t) += ::thrust::get<1>(t);
    }
};

struct MinusAssignmentFunctor
{
    template<class Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        ::thrust::get<0>(t) -= ::thrust::get<1>(t);
    }
};

struct TimesAssignmentFunctor
{
    template<class Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        ::thrust::get<0>(t) *= ::thrust::get<1>(t);
    }
};

struct DivideAssignmentFunctor
{
    template<class Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        ::thrust::get<0>(t) /= ::thrust::get<1>(t);
    }
};

template<typename T>
struct PlusValueAssignmentFunctor
{
    typedef T value_type;
    value_type value;
    PlusValueAssignmentFunctor(value_type value) : value(value) {}
    template<class Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        ::thrust::get<0>(t) += value;
    }
};

template<typename T>
struct MinusValueAssignmentFunctor
{
    typedef T value_type;
    value_type value;
    MinusValueAssignmentFunctor(value_type value) : value(value) {}
    template<class Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        ::thrust::get<0>(t) -= value;
    }
};

template<typename T>
struct TimesValueAssignmentFunctor
{
    typedef T value_type;
    value_type value;
    TimesValueAssignmentFunctor(value_type value) : value(value) {}
    template<class Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        ::thrust::get<0>(t) *= value;
    }
};

template<typename T>
struct DivideValueAssignmentFunctor
{
    typedef T value_type;
    value_type value;
    DivideValueAssignmentFunctor(value_type value) : value(value) {}
    template<class Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        ::thrust::get<0>(t) /= value;
    }
};

template<class LValue, class Expression>
void assign_in_place(LValue& lvalue, const Expression& expression)
{
    typedef ::thrust::tuple<typename LValue::iterator, typename Expression::const_iterator> tuple_type;

    ::thrust::for_each(
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.begin(), expression.begin())),
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.end(), expression.end())),
        AssignmentFunctor()
    );
}

template<class LValue, class Expression>
void assign_in_place_specialized(LValue& lvalue, const Expression& expression)
{
    std::cout << "specialized inplace assgnment" << std::endl;

    // have the expression implement it's own specialization
    expression.apply(lvalue);
}

template<class LValue, class Expression>
void plus_assign_in_place(LValue& lvalue, const Expression& expression)
{
    typedef ::thrust::tuple<typename LValue::iterator, typename Expression::const_iterator> tuple_type;

    ::thrust::for_each(
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.begin(), expression.begin())),
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.end(), expression.end())),
        PlusAssignmentFunctor()
    );
}

template<class LValue, class Expression>
void minus_assign_in_place(LValue& lvalue, const Expression& expression)
{
    typedef ::thrust::tuple<typename LValue::iterator, typename Expression::const_iterator> tuple_type;
    
    ::thrust::for_each(
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.begin(), expression.begin())),
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.end(), expression.end())),
        MinusAssignmentFunctor()
    );
}

template<class LValue, class Expression>
void times_assign_in_place(LValue& lvalue, const Expression& expression)
{
    typedef ::thrust::tuple<typename LValue::iterator, typename Expression::const_iterator> tuple_type;
    
    ::thrust::for_each(
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.begin(), expression.begin())),
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.end(), expression.end())),
        TimesAssignmentFunctor()
    );
}

template<class LValue, class Expression>
void divide_assign_in_place(LValue& lvalue, const Expression& expression)
{
    typedef ::thrust::tuple<typename LValue::iterator, typename Expression::const_iterator> tuple_type;
    
    ::thrust::for_each(
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.begin(), expression.begin())),
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.end(), expression.end())),
        DivideAssignmentFunctor()
    );
}

template<class LValue>
void assign_value_in_place(LValue& lvalue, const typename LValue::value_type value)
{
    thrust::fill(lvalue.begin(), lvalue.end(), value);
}

template<class LValue>
void plus_assign_value_in_place(LValue& lvalue, const typename LValue::value_type value)
{
    typedef typename LValue::value_type value_type;
    typedef ::thrust::tuple<typename LValue::iterator> tuple_type;
    ::thrust::for_each(
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.begin())),
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.end())),
        PlusValueAssignmentFunctor<value_type>(value)
    );
}

template<class LValue>
void minus_assign_value_in_place(LValue& lvalue, const typename LValue::value_type value)
{
    typedef typename LValue::value_type value_type;
    typedef ::thrust::tuple<typename LValue::iterator> tuple_type;
    ::thrust::for_each(
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.begin())),
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.end())),
        MinusValueAssignmentFunctor<value_type>(value)
    );
}

template<class LValue>
void times_assign_value_in_place(LValue& lvalue, const typename LValue::value_type value)
{
    typedef typename LValue::value_type value_type;
    typedef ::thrust::tuple<typename LValue::iterator> tuple_type;
    ::thrust::for_each(
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.begin())),
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.end())),
        TimesValueAssignmentFunctor<value_type>(value)
    );
}

template<class LValue>
void divide_assign_value_in_place(LValue& lvalue, const typename LValue::value_type value)
{
    typedef typename LValue::value_type value_type;
    typedef ::thrust::tuple<typename LValue::iterator> tuple_type;
    ::thrust::for_each(
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.begin())),
        ::thrust::zip_iterator<tuple_type>(::thrust::make_tuple(lvalue.end())),
        DivideValueAssignmentFunctor<value_type>(value)
    );
}

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
