/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef ASSIGN_DIMENSIONS_H
#define ASSIGN_DIMENSIONS_H

#include <cassert>

namespace minlin {
    
namespace detail {

// Helpers to assign dimensions in matrix/vector assignments.
// The rules are (where u,v are vectors, A,B are matrices, and lhs,rhs are anything):

// u = v transfers dimensions
// A = B transfers dimensions
// u = A transfers dimensions
// A = u transfers dimensions
// lhs(all) = rhs keeps dimensions as-is

// lhs(I) where I is an index vector is as per lhs(all)

template<bool InPlace>
struct assign_dimensions_to_v // <false>
{
    template<class Left, class Right>
    assign_dimensions_to_v(Left& left, const Right& right)
    {
        left.orientation_ = right.orientation();
    }
};

template<>
struct assign_dimensions_to_v<true>
{
    template<class Left, class Right>
    assign_dimensions_to_v(Left& left, const Right& right)
    {
        // Leave dimensions as-is for in-place operations
    }
};

template<bool InPlace>
struct assign_dimensions_to_m // <false>
{
    template<class Left, class Right>
    assign_dimensions_to_m(Left& left, const Right& right)
    {
        left.rows_ = right.rows();
        left.cols_ = right.cols();
    }
};

template<>
struct assign_dimensions_to_m<true>
{
    template<class Left, class Right>
    assign_dimensions_to_m(Left& left, const Right& right)
    {
        // Leave dimensions as-is for in-place operations
    }
};

} // end namespace detail

} // end namespace minlin

#endif
