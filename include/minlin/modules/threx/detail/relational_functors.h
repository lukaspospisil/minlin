/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_RELATIONAL_FUNCTORS_H
#define THREX_DETAIL_RELATIONAL_FUNCTORS_H

#include <thrust/tuple.h>

namespace minlin {

namespace threx {

namespace detail {

template<typename T>
struct SequalToVFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit SequalToVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return scalar == value;
    }
};

template<typename T>
struct SnotEqualToVFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit SnotEqualToVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return scalar != value;
    }
};

template<typename T>
struct SlessThanVFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit SlessThanVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return scalar < value;
    }
};

template<typename T>
struct SlessThanOrEqualToVFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit SlessThanOrEqualToVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return scalar <= value;
    }
};

template<typename T>
struct SgreaterThanVFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit SgreaterThanVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return scalar > value;
    }
};

template<typename T>
struct SgreaterThanOrEqualToVFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit SgreaterThanOrEqualToVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return scalar >= value;
    }
};

template<typename T>
struct VequalToSFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VequalToSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return value == scalar;
    }
};

template<typename T>
struct VnotEqualToSFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VnotEqualToSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return value != scalar;
    }
};

template<typename T>
struct VlessThanSFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VlessThanSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return value < scalar;
    }
};

template<typename T>
struct VlessThanOrEqualToSFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VlessThanOrEqualToSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return value <= scalar;
    }
};

template<typename T>
struct VgreaterThanSFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VgreaterThanSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return value > scalar;
    }
};

template<typename T>
struct VgreaterThanOrEqualToSFunctor {
	typedef bool result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VgreaterThanOrEqualToSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return value >= scalar;
    }
};

template<typename T>
struct VequalToVFunctor {
	typedef bool result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) == ::thrust::get<1>(t);
    }
};

template<typename T>
struct VnotEqualToVFunctor {
	typedef bool result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) != ::thrust::get<1>(t);
    }
};

template<typename T>
struct VlessThanVFunctor {
	typedef bool result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) < ::thrust::get<1>(t);
    }
};

template<typename T>
struct VlessThanOrEqualToVFunctor {
	typedef bool result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) <= ::thrust::get<1>(t);
    }
};

template<typename T>
struct VgreaterThanVFunctor {
	typedef bool result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) > ::thrust::get<1>(t);
    }
};

template<typename T>
struct VgreaterThanOrEqualToVFunctor {
	typedef bool result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) >= ::thrust::get<1>(t);
    }
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
