/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_HOST_VECTOR_H
#define THREX_HOST_VECTOR_H

#include "storage.h"

#include <minlin/vector.h>

#include <thrust/host_vector.h>

// Convenience header file for Thrust host vectors.

// Defines a convenience template HostVector which is a (derived type of)
// Vector that uses a thrust::host_vector as its storage.

// So, rather than Vector<ByValue<thrust::host_vector<T> > > v;
// simply use HostVector<T> v;

// This header file also pulls in all the functionality of host vector.
// If only a subset of functionality is desired (say, because the underlying
// type T only supports a subset of functionality) it might be desirable not
// to include this header file, and instead just the individual files desired.


namespace minlin {
    
namespace threx {

// No template typedefs in C++03, so use a derived class template instead
template<typename T>
class HostVector
    : public Vector<ByValue<thrust::host_vector<T> > > {
public:        
    typedef Vector<ByValue<thrust::host_vector<T> > > base;

    // Just need to write the constructors and assignment operators.
    // Everything else is inherited.

    HostVector() : base() {}

    explicit HostVector(typename base::difference_type n) : base(n) {}

    HostVector(const HostVector& other) : base(other) {}

    template<class OtherVector>
    HostVector(const OtherVector& other) : base(other) {}

    explicit HostVector(const typename base::expression_type& expression) : base(expression) {}
    
    HostVector& operator=(const HostVector& other)
    {
        base::operator=(other);
        return *this;
    }

    template<class OtherVector>
    HostVector& operator=(const OtherVector& other)
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
