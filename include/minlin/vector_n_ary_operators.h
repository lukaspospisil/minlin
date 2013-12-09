/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_VECTOR_N_ARY_OPERATORS_H
#define MINLIN_VECTOR_N_ARY_OPERATORS_H

#include "vector.h"

namespace minlin {

// Vector-Vector-...-Vector operators
// **********************************

template<class E1, class E2, class E3>
Vector<typename E1::template v_times_v3<E1, E2, E3>::type>
mul(const Vector<E1>& v1,
    const Vector<E2>& v2,
    const Vector<E3>& v3)
{
    #ifdef MINLIN_DEBUG
    assert(v1.rows() == v2.rows() && v1.cols() == v2.cols());
    assert(v1.rows() == v3.rows() && v1.cols() == v3.cols());
    #endif
    return make_vector(
        mul(
            v1.expression(),
            v2.expression(),
            v3.expression()
        ), v1.orientation()
    );
}

template<class E1, class E2, class E3, class E4>
Vector<typename E1::template v_times_v4<E1, E2, E3, E4>::type>
mul(const Vector<E1>& v1,
    const Vector<E2>& v2,
    const Vector<E3>& v3,
    const Vector<E4>& v4)
{
    #ifdef MINLIN_DEBUG
    assert(v1.rows() == v2.rows() && v1.cols() == v2.cols());
    assert(v1.rows() == v3.rows() && v1.cols() == v3.cols());
    assert(v1.rows() == v4.rows() && v1.cols() == v4.cols());
    #endif
    return make_vector(
        mul(
            v1.expression(),
            v2.expression(),
            v3.expression(),
            v4.expression()
        ), v1.orientation()
    );
}

template<class E1, class E2, class E3, class E4, class E5>
Vector<typename E1::template v_times_v5<E1, E2, E3, E4, E5>::type>
mul(const Vector<E1>& v1,
    const Vector<E2>& v2,
    const Vector<E3>& v3,
    const Vector<E4>& v4,
    const Vector<E5>& v5)
{
    #ifdef MINLIN_DEBUG
    assert(v1.rows() == v2.rows() && v1.cols() == v2.cols());
    assert(v1.rows() == v3.rows() && v1.cols() == v3.cols());
    assert(v1.rows() == v4.rows() && v1.cols() == v4.cols());
    assert(v1.rows() == v5.rows() && v1.cols() == v5.cols());
    #endif
    return make_vector(
        mul(
            v1.expression(),
            v2.expression(),
            v3.expression(),
            v4.expression(),
            v5.expression()
        ), v1.orientation()
    );
}

template<class E1, class E2, class E3, class E4, class E5,
         class E6>
Vector<typename E1::template v_times_v6<E1, E2, E3, E4, E5, E6>::type>
mul(const Vector<E1>& v1,
    const Vector<E2>& v2,
    const Vector<E3>& v3,
    const Vector<E4>& v4,
    const Vector<E5>& v5,
    const Vector<E6>& v6)
{
    #ifdef MINLIN_DEBUG
    assert(v1.rows() == v2.rows() && v1.cols() == v2.cols());
    assert(v1.rows() == v3.rows() && v1.cols() == v3.cols());
    assert(v1.rows() == v4.rows() && v1.cols() == v4.cols());
    assert(v1.rows() == v5.rows() && v1.cols() == v5.cols());
    assert(v1.rows() == v6.rows() && v1.cols() == v6.cols());
    #endif
    return make_vector(
        mul(
            v1.expression(),
            v2.expression(),
            v3.expression(),
            v4.expression(),
            v5.expression(),
            v6.expression()
        ), v1.orientation()
    );
}

template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7>
Vector<typename E1::template v_times_v7<E1, E2, E3, E4, E5, E6, E7>::type>
mul(const Vector<E1>& v1,
    const Vector<E2>& v2,
    const Vector<E3>& v3,
    const Vector<E4>& v4,
    const Vector<E5>& v5,
    const Vector<E6>& v6,
    const Vector<E7>& v7)
{
    #ifdef MINLIN_DEBUG
    assert(v1.rows() == v2.rows() && v1.cols() == v2.cols());
    assert(v1.rows() == v3.rows() && v1.cols() == v3.cols());
    assert(v1.rows() == v4.rows() && v1.cols() == v4.cols());
    assert(v1.rows() == v5.rows() && v1.cols() == v5.cols());
    assert(v1.rows() == v6.rows() && v1.cols() == v6.cols());
    assert(v1.rows() == v7.rows() && v1.cols() == v7.cols());
    #endif
    return make_vector(
        mul(
            v1.expression(),
            v2.expression(),
            v3.expression(),
            v4.expression(),
            v5.expression(),
            v6.expression(),
            v7.expression()
        ), v1.orientation()
    );
}

template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7, class E8>
Vector<typename E1::template v_times_v8<E1, E2, E3, E4, E5, E6, E7, E8>::type>
mul(const Vector<E1>& v1,
    const Vector<E2>& v2,
    const Vector<E3>& v3,
    const Vector<E4>& v4,
    const Vector<E5>& v5,
    const Vector<E6>& v6,
    const Vector<E7>& v7,
    const Vector<E8>& v8)
{
    #ifdef MINLIN_DEBUG
    assert(v1.rows() == v2.rows() && v1.cols() == v2.cols());
    assert(v1.rows() == v3.rows() && v1.cols() == v3.cols());
    assert(v1.rows() == v4.rows() && v1.cols() == v4.cols());
    assert(v1.rows() == v5.rows() && v1.cols() == v5.cols());
    assert(v1.rows() == v6.rows() && v1.cols() == v6.cols());
    assert(v1.rows() == v7.rows() && v1.cols() == v7.cols());
    assert(v1.rows() == v8.rows() && v1.cols() == v8.cols());
    #endif
    return make_vector(
        mul(
            v1.expression(),
            v2.expression(),
            v3.expression(),
            v4.expression(),
            v5.expression(),
            v6.expression(),
            v7.expression(),
            v8.expression()
        ), v1.orientation()
    );
}

template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7, class E8, class E9>
Vector<typename E1::template v_times_v9<E1, E2, E3, E4, E5, E6, E7, E8, E9>::type>
mul(const Vector<E1>& v1,
    const Vector<E2>& v2,
    const Vector<E3>& v3,
    const Vector<E4>& v4,
    const Vector<E5>& v5,
    const Vector<E6>& v6,
    const Vector<E7>& v7,
    const Vector<E8>& v8,
    const Vector<E9>& v9)
{
    #ifdef MINLIN_DEBUG
    assert(v1.rows() == v2.rows() && v1.cols() == v2.cols());
    assert(v1.rows() == v3.rows() && v1.cols() == v3.cols());
    assert(v1.rows() == v4.rows() && v1.cols() == v4.cols());
    assert(v1.rows() == v5.rows() && v1.cols() == v5.cols());
    assert(v1.rows() == v6.rows() && v1.cols() == v6.cols());
    assert(v1.rows() == v7.rows() && v1.cols() == v7.cols());
    assert(v1.rows() == v8.rows() && v1.cols() == v8.cols());
    assert(v1.rows() == v9.rows() && v1.cols() == v9.cols());
    #endif
    return make_vector(
        mul(
            v1.expression(),
            v2.expression(),
            v3.expression(),
            v4.expression(),
            v5.expression(),
            v6.expression(),
            v7.expression(),
            v8.expression(),
            v9.expression()
        ), v1.orientation()
    );
}

template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7, class E8, class E9, class E10>
Vector<typename E1::template v_times_v10<E1, E2, E3, E4, E5, E6, E7, E8, E9, E10>::type>
mul(const Vector<E1>& v1,
    const Vector<E2>& v2,
    const Vector<E3>& v3,
    const Vector<E4>& v4,
    const Vector<E5>& v5,
    const Vector<E6>& v6,
    const Vector<E7>& v7,
    const Vector<E8>& v8,
    const Vector<E9>& v9,
    const Vector<E10>& v10)
{
    #ifdef MINLIN_DEBUG
    assert(v1.rows() == v2.rows() && v1.cols() == v2.cols());
    assert(v1.rows() == v3.rows() && v1.cols() == v3.cols());
    assert(v1.rows() == v4.rows() && v1.cols() == v4.cols());
    assert(v1.rows() == v5.rows() && v1.cols() == v5.cols());
    assert(v1.rows() == v6.rows() && v1.cols() == v6.cols());
    assert(v1.rows() == v7.rows() && v1.cols() == v7.cols());
    assert(v1.rows() == v8.rows() && v1.cols() == v8.cols());
    assert(v1.rows() == v9.rows() && v1.cols() == v9.cols());
    assert(v1.rows() == v10.rows() && v1.cols() == v10.cols());
    #endif
    return make_vector(
        mul(
            v1.expression(),
            v2.expression(),
            v3.expression(),
            v4.expression(),
            v5.expression(),
            v6.expression(),
            v7.expression(),
            v8.expression(),
            v9.expression(),
            v10.expression()
        ), v1.orientation()
    );
}

} // end namespace minlin

#endif
