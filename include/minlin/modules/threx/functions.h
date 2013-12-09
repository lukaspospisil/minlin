/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_FUNCTIONS_H
#define THREX_FUNCTIONS_H

#include "detail/function_functors.h"
#include "storage.h"

#include <minlin/vector.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/functional.h>

#include <algorithm>

namespace minlin {

namespace threx {

// Any
// ***
template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
bool>::type
any_of(const Expression& expression)
{
    return thrust::any_of(expression.begin(), expression.end(), thrust::identity<bool>());
}

// All
// ***
template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
bool>::type
all_of(const Expression& expression)
{
    return thrust::all_of(expression.begin(), expression.end(), thrust::identity<bool>());
}

// Repvec
// ******
template<typename T>
Vector<Range<thrust::constant_iterator<T> > >
repvec(T value, typename Range<thrust::constant_iterator<T> >::difference_type n)
{
    thrust::constant_iterator<T> begin(value);
    Range<thrust::constant_iterator<T> > range_expression(begin, begin + n);
    return make_vector(range_expression, RowOriented);
}

// Zeros
// *****
template<typename T>
Vector<Range<thrust::constant_iterator<T> > >
zeros(typename Range<thrust::constant_iterator<T> >::difference_type n)
{
    return repvec<T>(T(), n);
}

inline
Vector<Range<thrust::constant_iterator<int> > >
zeros(typename Range<thrust::constant_iterator<int> >::difference_type n)
{
    return zeros<int>(n);
}

// Ones
// ****
template<typename T>
Vector<Range<thrust::constant_iterator<T> > >
ones(typename Range<thrust::constant_iterator<T> >::difference_type n)
{
    return repvec<T>(1, n);
}

inline
Vector<Range<thrust::constant_iterator<int> > >
ones(typename Range<thrust::constant_iterator<int> >::difference_type n)
{
    return ones<int>(n);
}

// Range
// *****
template<typename T>
Vector<Range<thrust::counting_iterator<T> > >
range(T a, T b)
{
    typedef thrust::counting_iterator<T> iterator_type;
    typedef Range<iterator_type> range_type;
    typedef typename range_type::difference_type difference_type;

    iterator_type begin(a);
    difference_type size = std::max(0, b-a+1);
    return make_vector(range_type(begin, begin + size), RowOriented);
}

template<typename T>
Vector<Range<thrust::transform_iterator<detail::RangeFunctor<T>, thrust::counting_iterator<T> > > >
range(T a, T c, T b)
{
    typedef thrust::transform_iterator<
        detail::RangeFunctor<T>,
        thrust::counting_iterator<T>
    > iterator_type;

    typedef Range<iterator_type> range_type;
    typedef typename range_type::difference_type difference_type;

    thrust::counting_iterator<T> begin(0);
    iterator_type transformed_begin(begin, detail::RangeFunctor<T>(a, c));
    difference_type size = difference_type((b-a)/c+1);
    size = std::max(difference_type(0), size);
    return make_vector(range_type(transformed_begin, transformed_begin + size), RowOriented);
}

// Rand
// ****
template<typename T>
Vector<
    Range<
        thrust::transform_iterator<
            detail::RandomFunctor<T>,
            thrust::constant_iterator<T>
        >
    >
>
rand(typename Range<thrust::transform_iterator<
                    detail::RandomFunctor<T>, thrust::constant_iterator<T>
                 > >::difference_type n)
{
    typedef thrust::constant_iterator<T> dummy_iterator;
    typedef thrust::transform_iterator<detail::RandomFunctor<T>, dummy_iterator> iterator;
    iterator begin(dummy_iterator(0), detail::RandomFunctor<T>());
    Range<iterator> range_expression(begin, begin + n);
    return make_vector(range_expression, RowOriented);
}

} // end namespace threx

} // end namespace minlin

#endif
