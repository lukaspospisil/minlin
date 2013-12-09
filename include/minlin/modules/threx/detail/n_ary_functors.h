/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_N_ARY_FUNCTORS_H
#define THREX_DETAIL_N_ARY_FUNCTORS_H

#include <thrust/tuple.h>

#include <cmath>

namespace minlin {

namespace threx {

namespace detail {

// Vector-Vector multiplication
template<typename T>
struct VtimesVFunctor3 {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) * ::thrust::get<1>(t) * ::thrust::get<2>(t);
    }
};

template<typename T>
struct VtimesVFunctor4 {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) * ::thrust::get<1>(t) * ::thrust::get<2>(t) * ::thrust::get<3>(t);
    }
};

template<typename T>
struct VtimesVFunctor5 {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) * ::thrust::get<1>(t) * ::thrust::get<2>(t) * ::thrust::get<3>(t) * ::thrust::get<4>(t);
    }
};

template<typename T>
struct VtimesVFunctor6 {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) * ::thrust::get<1>(t) * ::thrust::get<2>(t) * ::thrust::get<3>(t) * ::thrust::get<4>(t) * ::thrust::get<5>(t);
    }
};

template<typename T>
struct VtimesVFunctor7 {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) * ::thrust::get<1>(t) * ::thrust::get<2>(t) * ::thrust::get<3>(t) * ::thrust::get<4>(t) * ::thrust::get<5>(t) * ::thrust::get<6>(t);
    }
};

template<typename T>
struct VtimesVFunctor8 {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) * ::thrust::get<1>(t) * ::thrust::get<2>(t) * ::thrust::get<3>(t) * ::thrust::get<4>(t) * ::thrust::get<5>(t) * ::thrust::get<6>(t) * ::thrust::get<7>(t);
    }
};

template<typename T>
struct VtimesVFunctor9 {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) * ::thrust::get<1>(t) * ::thrust::get<2>(t) * ::thrust::get<3>(t) * ::thrust::get<4>(t) * ::thrust::get<5>(t) * ::thrust::get<6>(t) * ::thrust::get<7>(t) * ::thrust::get<8>(t);
    }
};

template<typename T>
struct VtimesVFunctor10 {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) * ::thrust::get<1>(t) * ::thrust::get<2>(t) * ::thrust::get<3>(t) * ::thrust::get<4>(t) * ::thrust::get<5>(t) * ::thrust::get<6>(t) * ::thrust::get<7>(t) * ::thrust::get<8>(t) * ::thrust::get<9>(t);
    }
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
