/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_STORAGE_H
#define THREX_DETAIL_STORAGE_H

#include "inplace.h"
#include "assignment.h"

#include <iterator>

namespace minlin {

namespace threx {

namespace detail {

// Wraps an iterator range
template<typename Iterator>
class RangeWrapper {
public:

    typedef Iterator iterator;
    typedef iterator const_iterator;

    typedef typename std::iterator_traits<iterator>::value_type value_type;
    typedef typename std::iterator_traits<iterator>::difference_type difference_type;
    typedef typename std::iterator_traits<iterator>::reference reference;
    typedef reference const_reference;
    typedef typename std::iterator_traits<iterator>::pointer pointer;
    typedef pointer const_pointer;

    RangeWrapper(iterator first, iterator last) : first(first), last(last) {}

    iterator begin() const
    {
        return first;
    }

    iterator end() const
    {
        return last;
    }

    reference operator[](difference_type i)
    {
        return *(begin() + i);
    }

    value_type operator[](difference_type i) const
    {
        return *(begin() + i);
    }

    difference_type size() const
    {
        return end() - begin();
    }

private:
    iterator first, last;
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
