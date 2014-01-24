/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_UNARY_FUNCTORS_H
#define THREX_DETAIL_UNARY_FUNCTORS_H

#include <cmath>

namespace minlin {

namespace threx {
    
namespace detail {
#define THREX_DEFINE_UNARY_FUNCTOR(op) \
template<typename T> \
struct op##VFunctor { \
    typedef T result_type; \
    typedef T value_type; \
    __host__ __device__ \
    value_type operator()(value_type value) const \
    { \
        using std::op; \
        return op(value); \
    } \
};

THREX_DEFINE_UNARY_FUNCTOR(abs)
THREX_DEFINE_UNARY_FUNCTOR(acos)
THREX_DEFINE_UNARY_FUNCTOR(asin)
THREX_DEFINE_UNARY_FUNCTOR(atan)
THREX_DEFINE_UNARY_FUNCTOR(ceil)
THREX_DEFINE_UNARY_FUNCTOR(cos)
THREX_DEFINE_UNARY_FUNCTOR(cosh)
THREX_DEFINE_UNARY_FUNCTOR(exp)
THREX_DEFINE_UNARY_FUNCTOR(floor)
THREX_DEFINE_UNARY_FUNCTOR(log)
THREX_DEFINE_UNARY_FUNCTOR(log10)
THREX_DEFINE_UNARY_FUNCTOR(sin)
THREX_DEFINE_UNARY_FUNCTOR(sinh)
THREX_DEFINE_UNARY_FUNCTOR(sqrt)
THREX_DEFINE_UNARY_FUNCTOR(tan)
THREX_DEFINE_UNARY_FUNCTOR(tanh)

template<typename T>
struct plusVFunctor {
    typedef T result_type;
    typedef T value_type;
    __host__ __device__
    value_type operator()(value_type value) const
    {
        return +value;
    }
};

template<typename T>
struct minusVFunctor {
    typedef T result_type;
    typedef T value_type;
    __host__ __device__
    value_type operator()(value_type value) const
    {
        return -value;
    }
};

template<typename T>
struct powVFunctor {
    typedef T result_type;
    typedef T value_type;
    value_type scalar;

    explicit powVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    value_type operator()(value_type value) const
    {
        using std::pow;
        return pow(value, scalar);
    }
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
