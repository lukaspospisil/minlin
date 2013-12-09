/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_MATRIX_N_ARY_OPERATORS_H
#define MINLIN_MATRIX_N_ARY_OPERATORS_H

#include "matrix.h"

namespace minlin {

// Matrix-Matrix-...-Matrix operators
// **********************************

template<class E1, class E2, class E3>
Matrix<typename E1::template v_times_v3<E1, E2, E3>::type>
mul(const Matrix<E1>& m1,
    const Matrix<E2>& m2,
    const Matrix<E3>& m3)
{
    #ifdef MINLIN_DEBUG
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    assert(m1.rows() == m3.rows() && m1.cols() == m3.cols());
    #endif
    return make_matrix(
        mul(
            m1.expression(),
            m2.expression(),
            m3.expression()
        ), m1.rows(), m1.cols()
    );
}

template<class E1, class E2, class E3, class E4>
Matrix<typename E1::template v_times_v4<E1, E2, E3, E4>::type>
mul(const Matrix<E1>& m1,
    const Matrix<E2>& m2,
    const Matrix<E3>& m3,
    const Matrix<E4>& m4)
{
    #ifdef MINLIN_DEBUG
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    assert(m1.rows() == m3.rows() && m1.cols() == m3.cols());
    assert(m1.rows() == m4.rows() && m1.cols() == m4.cols());
    #endif
    return make_matrix(
        mul(
            m1.expression(),
            m2.expression(),
            m3.expression(),
            m4.expression()
        ), m1.rows(), m1.cols()
    );
}

template<class E1, class E2, class E3, class E4, class E5>
Matrix<typename E1::template v_times_v5<E1, E2, E3, E4, E5>::type>
mul(const Matrix<E1>& m1,
    const Matrix<E2>& m2,
    const Matrix<E3>& m3,
    const Matrix<E4>& m4,
    const Matrix<E5>& m5)
{
    #ifdef MINLIN_DEBUG
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    assert(m1.rows() == m3.rows() && m1.cols() == m3.cols());
    assert(m1.rows() == m4.rows() && m1.cols() == m4.cols());
    assert(m1.rows() == m5.rows() && m1.cols() == m5.cols());
    #endif
    return make_matrix(
        mul(
            m1.expression(),
            m2.expression(),
            m3.expression(),
            m4.expression(),
            m5.expression()
        ), m1.rows(), m1.cols()
    );
}

template<class E1, class E2, class E3, class E4, class E5,
         class E6>
Matrix<typename E1::template v_times_v6<E1, E2, E3, E4, E5, E6>::type>
mul(const Matrix<E1>& m1,
    const Matrix<E2>& m2,
    const Matrix<E3>& m3,
    const Matrix<E4>& m4,
    const Matrix<E5>& m5,
    const Matrix<E6>& m6)
{
    #ifdef MINLIN_DEBUG
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    assert(m1.rows() == m3.rows() && m1.cols() == m3.cols());
    assert(m1.rows() == m4.rows() && m1.cols() == m4.cols());
    assert(m1.rows() == m5.rows() && m1.cols() == m5.cols());
    assert(m1.rows() == m6.rows() && m1.cols() == m6.cols());
    #endif
    return make_matrix(
        mul(
            m1.expression(),
            m2.expression(),
            m3.expression(),
            m4.expression(),
            m5.expression(),
            m6.expression()
        ), m1.rows(), m1.cols()
    );
}

template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7>
Matrix<typename E1::template v_times_v7<E1, E2, E3, E4, E5, E6, E7>::type>
mul(const Matrix<E1>& m1,
    const Matrix<E2>& m2,
    const Matrix<E3>& m3,
    const Matrix<E4>& m4,
    const Matrix<E5>& m5,
    const Matrix<E6>& m6,
    const Matrix<E7>& m7)
{
    #ifdef MINLIN_DEBUG
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    assert(m1.rows() == m3.rows() && m1.cols() == m3.cols());
    assert(m1.rows() == m4.rows() && m1.cols() == m4.cols());
    assert(m1.rows() == m5.rows() && m1.cols() == m5.cols());
    assert(m1.rows() == m6.rows() && m1.cols() == m6.cols());
    assert(m1.rows() == m7.rows() && m1.cols() == m7.cols());
    #endif
    return make_matrix(
        mul(
            m1.expression(),
            m2.expression(),
            m3.expression(),
            m4.expression(),
            m5.expression(),
            m6.expression(),
            m7.expression()
        ), m1.rows(), m1.cols()
    );
}

template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7, class E8>
Matrix<typename E1::template v_times_v8<E1, E2, E3, E4, E5, E6, E7, E8>::type>
mul(const Matrix<E1>& m1,
    const Matrix<E2>& m2,
    const Matrix<E3>& m3,
    const Matrix<E4>& m4,
    const Matrix<E5>& m5,
    const Matrix<E6>& m6,
    const Matrix<E7>& m7,
    const Matrix<E8>& m8)
{
    #ifdef MINLIN_DEBUG
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    assert(m1.rows() == m3.rows() && m1.cols() == m3.cols());
    assert(m1.rows() == m4.rows() && m1.cols() == m4.cols());
    assert(m1.rows() == m5.rows() && m1.cols() == m5.cols());
    assert(m1.rows() == m6.rows() && m1.cols() == m6.cols());
    assert(m1.rows() == m7.rows() && m1.cols() == m7.cols());
    assert(m1.rows() == m8.rows() && m1.cols() == m8.cols());
    #endif
    return make_matrix(
        mul(
            m1.expression(),
            m2.expression(),
            m3.expression(),
            m4.expression(),
            m5.expression(),
            m6.expression(),
            m7.expression(),
            m8.expression()
        ), m1.rows(), m1.cols()
    );
}

template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7, class E8, class E9>
Matrix<typename E1::template v_times_v9<E1, E2, E3, E4, E5, E6, E7, E8, E9>::type>
mul(const Matrix<E1>& m1,
    const Matrix<E2>& m2,
    const Matrix<E3>& m3,
    const Matrix<E4>& m4,
    const Matrix<E5>& m5,
    const Matrix<E6>& m6,
    const Matrix<E7>& m7,
    const Matrix<E8>& m8,
    const Matrix<E9>& m9)
{
    #ifdef MINLIN_DEBUG
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    assert(m1.rows() == m3.rows() && m1.cols() == m3.cols());
    assert(m1.rows() == m4.rows() && m1.cols() == m4.cols());
    assert(m1.rows() == m5.rows() && m1.cols() == m5.cols());
    assert(m1.rows() == m6.rows() && m1.cols() == m6.cols());
    assert(m1.rows() == m7.rows() && m1.cols() == m7.cols());
    assert(m1.rows() == m8.rows() && m1.cols() == m8.cols());
    assert(m1.rows() == m9.rows() && m1.cols() == m9.cols());
    #endif
    return make_matrix(
        mul(
            m1.expression(),
            m2.expression(),
            m3.expression(),
            m4.expression(),
            m5.expression(),
            m6.expression(),
            m7.expression(),
            m8.expression(),
            m9.expression()
        ), m1.rows(), m1.cols()
    );
}

template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7, class E8, class E9, class E10>
Matrix<typename E1::template v_times_v10<E1, E2, E3, E4, E5, E6, E7, E8, E9, E10>::type>
mul(const Matrix<E1>& m1,
    const Matrix<E2>& m2,
    const Matrix<E3>& m3,
    const Matrix<E4>& m4,
    const Matrix<E5>& m5,
    const Matrix<E6>& m6,
    const Matrix<E7>& m7,
    const Matrix<E8>& m8,
    const Matrix<E9>& m9,
    const Matrix<E10>& v10)
{
    #ifdef MINLIN_DEBUG
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    assert(m1.rows() == m3.rows() && m1.cols() == m3.cols());
    assert(m1.rows() == m4.rows() && m1.cols() == m4.cols());
    assert(m1.rows() == m5.rows() && m1.cols() == m5.cols());
    assert(m1.rows() == m6.rows() && m1.cols() == m6.cols());
    assert(m1.rows() == m7.rows() && m1.cols() == m7.cols());
    assert(m1.rows() == m8.rows() && m1.cols() == m8.cols());
    assert(m1.rows() == m9.rows() && m1.cols() == m9.cols());
    assert(m1.rows() == v10.rows() && m1.cols() == v10.cols());
    #endif
    return make_matrix(
        mul(
            m1.expression(),
            m2.expression(),
            m3.expression(),
            m4.expression(),
            m5.expression(),
            m6.expression(),
            m7.expression(),
            m8.expression(),
            m9.expression(),
            v10.expression()
        ), m1.rows(), m1.cols()
    );
}

} // end namespace minlin

#endif
