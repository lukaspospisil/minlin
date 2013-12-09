/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_BINARY_FUNCTORS_H
#define THREX_DETAIL_BINARY_FUNCTORS_H

#include <thrust/tuple.h>

#include <cmath>

namespace minlin {

namespace threx {

namespace detail {

// Scalar-Vector addition
template<typename T>
struct SplusVFunctor {
	typedef T result_type;
    typedef T value_type;
    value_type scalar;    

    explicit SplusVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return scalar + value;
    }
};

// Vector-Scalar addition
template<typename T>
struct VplusSFunctor {
	typedef T result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VplusSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return value + scalar;
    }
};

// Scalar-Vector subtraction
template<typename T>
struct SminusVFunctor {
	typedef T result_type;
    typedef T value_type;
    value_type scalar;    

    explicit SminusVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return scalar - value;
    }
};

// Vector-Scalar subtraction
template<typename T>
struct VminusSFunctor {
	typedef T result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VminusSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return value - scalar;
    }
};

// Scalar-Vector multliplication
template<typename T>
struct StimesVFunctor {
	typedef T result_type;
    typedef T value_type;
    value_type scalar;    

    explicit StimesVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return scalar * value;
    }
};

// Vector-Scalar multliplication
template<typename T>
struct VtimesSFunctor {
	typedef T result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VtimesSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return value * scalar;
    }
};

// Scalar-Vector division
template<typename T>
struct SdivideVFunctor {
	typedef T result_type;
    typedef T value_type;
    value_type scalar;    

    explicit SdivideVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return scalar / value;
    }
};

// Vector-Scalar division
template<typename T>
struct VdivideSFunctor {
	typedef T result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VdivideSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        return value / scalar;
    }
};

// Scalar-Vector exponentiation
template<typename T>
struct SpowerVFunctor {
	typedef T result_type;
    typedef T value_type;
    value_type scalar;    

    explicit SpowerVFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        using std::pow;
        return pow(scalar, value);
    }
};

// Vector-Scalar exponentiation
template<typename T>
struct VpowerSFunctor {
	typedef T result_type;
    typedef T value_type;
    value_type scalar;    

    explicit VpowerSFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        using std::pow;
        return pow(value, scalar);
    }
};

// Scalar-Vector arctangent
template<typename T>
struct Satan2VFunctor {
	typedef T result_type;
    typedef T value_type;
    value_type scalar;    

    explicit Satan2VFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        using std::atan2;
        return atan2(scalar, value);
    }
};

// Vector-Scalar arctangent
template<typename T>
struct Vatan2SFunctor {
	typedef T result_type;
    typedef T value_type;
    value_type scalar;    

    explicit Vatan2SFunctor(value_type scalar) : scalar(scalar) {}

    __host__ __device__
    result_type operator()(value_type value) const
    {
        using std::atan2;
        return atan2(value, scalar);
    }
};

// Vector-Vector addition
template<typename T>
struct VplusVFunctor {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) + ::thrust::get<1>(t);
    }
};

// Vector-Vector subtraction
template<typename T>
struct VminusVFunctor {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) - ::thrust::get<1>(t);
    }
};

// Vector-Vector multiplication
template<typename T>
struct VtimesVFunctor {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) * ::thrust::get<1>(t);
    }
};

// Vector-Vector division
template<typename T>
struct VdivideVFunctor {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        return ::thrust::get<0>(t) / ::thrust::get<1>(t);
    }
};

// Vector-Vector exponentiation
template<typename T>
struct VpowerVFunctor {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        using std::pow;
        return pow(::thrust::get<0>(t), ::thrust::get<1>(t));
    }
};

// Vector-Vector arctangent
template<typename T>
struct Vatan2VFunctor {
	typedef T result_type;
	template<class Tuple>
    __host__ __device__
    result_type operator()(Tuple t) const
    {
        using std::atan2;
        return atan2(::thrust::get<0>(t), ::thrust::get<1>(t));
    }
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
