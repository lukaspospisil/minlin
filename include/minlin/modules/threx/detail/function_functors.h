/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_FUNCTION_FUNCTORS_H
#define THREX_DETAIL_FUNCTION_FUNCTORS_H

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

namespace minlin {

namespace threx {

namespace detail {

template<typename T>
struct RangeFunctor {
    typedef T value_type;
    typedef T result_type;
    value_type shift;
    value_type scale;
    RangeFunctor(value_type shift, value_type scale) : shift(shift), scale(scale) {}
    __host__ __device__
    result_type operator()(value_type value) const
    {
        return shift + scale * value;
    }
};

template<typename T>
struct RandomFunctorBase
{
    // Use the template mechanism to ensure exactly one instance per type
    static thrust::minstd_rand0 rng;
};

template<typename T>
thrust::minstd_rand0 RandomFunctorBase<T>::rng;

template<typename T>
struct RandomFunctor {};

template<>
struct RandomFunctor<double> : public RandomFunctorBase<double> {
    typedef double value_type;
    typedef double result_type;

    thrust::uniform_real_distribution<double> dist;
    RandomFunctor() : dist(0,1) {}
    __host__ __device__
    result_type operator()(value_type dummy)
    {
        //return dist(rng);
        return result_type(0.5);
    }
};


} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
