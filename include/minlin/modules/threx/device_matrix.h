/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DEVICE_MATRIX_H
#define THREX_DEVICE_MATRIX_H

#include "storage.h"

#include <minlin/matrix.h>

#include <thrust/device_vector.h>

// Convenience header file for Thrust device matrices.

// Defines a convenience template DeviceMatrix which is a (derived type of)
// Matrix that uses a thrust::device_vector as its storage.

// See host_vector.h for more documentation.

namespace minlin {

namespace threx {

template<typename T>
class DeviceMatrix
    : public Matrix<ByValue<thrust::device_vector<T> > > {
public:
    typedef Matrix<ByValue<thrust::device_vector<T> > > base;

    DeviceMatrix() : base() {}

    DeviceMatrix(typename base::difference_type m, typename base::difference_type n) : base(m, n) {}

    DeviceMatrix(const DeviceMatrix& other) : base(other) {}

    template<class OtherMatrix>
    DeviceMatrix(const OtherMatrix& other) : base(other) {}

    explicit DeviceMatrix(const typename base::expression_type& expression) : base(expression) {}

    DeviceMatrix& operator=(const DeviceMatrix& other)
    {
        base::operator=(other);
        return *this;
    }

    template<class OtherMatrix>
    DeviceMatrix& operator=(const OtherMatrix& other)
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
