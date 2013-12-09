/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_LOGICAL_FUNCTORS_H
#define THREX_DETAIL_LOGICAL_FUNCTORS_H

#include <thrust/tuple.h>

namespace minlin {

namespace threx {

namespace detail {

template<typename T>
struct SorVFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit SorVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return scalar || value;
    }
};

template<typename T>
struct SandVFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit SandVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return scalar && value;
    }
};

template<typename T>
struct VorSFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VorSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return value || scalar;
    }
};

template<typename T>
struct VandSFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VandSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return value && scalar;
    }
};

template<typename T>
struct VorVFunctor {
	typedef bool result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) || ::thrust::get<1>(t);
    }
};

template<typename T>
struct VandVFunctor {
	typedef bool result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) && ::thrust::get<1>(t);
    }
};

template<typename T>
struct notVFunctor {
    typedef bool result_type;
    typedef T value_type;
    __host__ __device__
    value_type operator()(value_type value) const
    {
        return !value;
    }
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
