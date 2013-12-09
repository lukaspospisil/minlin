/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef ASSERT_CONFORMANCE_H
#define ASSERT_CONFORMANCE_H

#include <cassert>

namespace minlin {
    
namespace detail {

// Helpers to check conformance of assignment and compound assignment.
// The rules are (where u,v are vectors, A,B are matrices, and lhs,rhs are anything):

// u = v is always allowed
// A = B is always allowed
// u = A requires A to have a singleton dimension
// A = u is always allowed
// lhs ?= rhs requires lhs and rhs to have the same dimensions
// lhs(all)  = rhs requires lhs and rhs to have the same size
// lhs(all) ?= rhs requires lhs and rhs to have the same size

// lhs(I) where I is an index vector is as per lhs(all)

template<bool InPlace>
struct assert_conformance // <false>
{
    template<class Left, class Right>
    static void assignment(const Left& left, Right& right)
    {
        // nothing to assert (allocating assignment)
    }

    template<class Left, class Right>
    static void assignment_m_to_v(const Left& left, Right& right)
    {
        assert(right.rows() == 1 || right.cols() == 1);
    }

    template<class Left, class Right>
    static void compound(const Left& left, Right& right)
    {
        assert(left.rows() == right.rows() && left.cols() == right.cols());
    }
};

template<>
struct assert_conformance<true> {

    template<class Left, class Right>
    static void assignment(const Left& left, Right& right)
    {
        assert(left.size() == right.size());
    }

    template<class Left, class Right>
    static void assignment_m_to_v(const Left& left, Right& right)
    {
        assert(left.size() == right.size());
    }

    template<class Left, class Right>
    static void compound(const Left& left, Right& right)
    {
        assert(left.size() == right.size());
    }
};

} // end namespace detail

} // end namespace minlin

#endif
