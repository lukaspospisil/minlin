/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_DETAIL_END_AND_ALL_H
#define MINLIN_DETAIL_END_AND_ALL_H

#include <limits>

// Todo: rename this file and/or organise these things more cleanly

namespace minlin {

// Double precision infinity value
namespace { double inf = std::numeric_limits<double>::infinity(); }
//  Todo: handle not being present / document that it must be

// For indexing expressions of the form u(end), A(1, end), etc.
struct end_type {};
namespace { end_type end; }

// For indexing expressions of the form u(all), A(1, all), etc.
// This is how we'd *like* to do it.

//struct all_type {};
//namespace { all_type all; }

// But some CUDA header seems to want to define a bool all(bool) in global scope.
// Unbelievable!  So here's the workaround for now - define our all as the same type.

namespace detail {
    typedef bool (all_type)(bool);
} // end namespace detail

inline bool all(bool p) { return p; }   // dummy function called all

} // end namespace minlin

#endif
