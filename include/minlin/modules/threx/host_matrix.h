/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_HOST_MATRIX_H
#define THREX_HOST_MATRIX_H

#include "storage.h"

#include <minlin/matrix.h>

#include <thrust/host_vector.h>

// Convenience header file for Thrust host matrices.

// Defines a convenience template HostMatrix which is a (derived type of)
// Matrix that uses a thrust::host_vector as its storage.

// See host_vector.h for more documentation.

namespace minlin {
    
namespace threx {

template<typename T>
class HostMatrix
    : public Matrix<ByValue<thrust::host_vector<T> > > {
public:        
    typedef Matrix<ByValue<thrust::host_vector<T> > > base;

    HostMatrix() : base() {}

    HostMatrix(typename base::difference_type m, typename base::difference_type n) : base(m, n) {}

    HostMatrix(const HostMatrix& other) : base(other) {}

    template<class OtherMatrix>
    HostMatrix(const OtherMatrix& other) : base(other) {}

    explicit HostMatrix(const typename base::expression_type& expression) : base(expression) {}

    HostMatrix& operator=(const HostMatrix& other)
    {
        base::operator=(other);
        return *this;
    }

    template<class OtherMatrix>
    HostMatrix& operator=(const OtherMatrix& other)
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
