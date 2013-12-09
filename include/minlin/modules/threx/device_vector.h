/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DEVICE_VECTOR_H
#define THREX_DEVICE_VECTOR_H

#include "storage.h"

#include <minlin/vector.h>

#include <thrust/device_vector.h>

// Convenience header file for Thrust host vectors.

// Defines a convenience template DeviceVector which is a (derived type of)
// Vector that uses a thrust::host_vector as its storage.

// See host_vector.h for more documentation.

namespace minlin {

namespace threx {

template<typename T>
class DeviceVector
    : public Vector<ByValue<thrust::device_vector<T> > > {
public:
    typedef Vector<ByValue<thrust::device_vector<T> > > base;

    DeviceVector() : base() {}

    explicit DeviceVector(typename base::difference_type n) : base(n) {}

    DeviceVector(const DeviceVector& other) : base(other) {}

    template<class OtherVector>
    DeviceVector(const OtherVector& other) : base(other) {}

    explicit DeviceVector(const typename base::expression_type& expression) : base(expression) {}

    DeviceVector& operator=(const DeviceVector& other)
    {
        base::operator=(other);
        return *this;
    }

    template<class OtherVector>
    DeviceVector& operator=(const OtherVector& other)
    {
        base::operator=(other);
        return *this;
    }

    // return raw pointer to start of data
    T* pointer()
    {
        return thrust::raw_pointer_cast( base::expression().data() );
    }

    // return raw pointer to start of data
    const T* pointer() const
    {
        return thrust::raw_pointer_cast( base::expression().data() );
    }
};

} // end namespace threx

} // end namespace minlin

#endif
