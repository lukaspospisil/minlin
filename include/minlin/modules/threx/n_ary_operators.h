/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_N_ARY_OPERATORS_H
#define THREX_N_ARY_OPERATORS_H

#include "detail/expression_types.h"
#include "detail/n_ary_expressions.h"

namespace minlin {

namespace threx {

template<class E1, class E2, class E3>
typename detail::expression_enabler<E1::is_expression,  // enabler on just the first argument should be fine
detail::VtimesV3<const E1, const E2, const E3> >::type
mul(const E1& e1, const E2& e2, const E3& e3)
{
    return detail::VtimesV3<const E1, const E2, const E3>
    (e1, e2, e3);
}

template<class E1, class E2, class E3, class E4>
typename detail::expression_enabler<E1::is_expression,  // enabler on just the first argument should be fine
detail::VtimesV4<const E1, const E2, const E3, const E4> >::type
mul(const E1& e1, const E2& e2, const E3& e3, const E4& e4)
{
    return detail::VtimesV4<const E1, const E2, const E3, const E4>
    (e1, e2, e3, e4);
}

template<class E1, class E2, class E3, class E4, class E5>
typename detail::expression_enabler<E1::is_expression,  // enabler on just the first argument should be fine
detail::VtimesV5<const E1, const E2, const E3, const E4, const E5> >::type
mul(const E1& e1, const E2& e2, const E3& e3, const E4& e4, const E5& e5)
{
    return detail::VtimesV5<const E1, const E2, const E3, const E4, const E5>
    (e1, e2, e3, e4, e5);
}

template<class E1, class E2, class E3, class E4, class E5, class E6>
typename detail::expression_enabler<E1::is_expression,  // enabler on just the first argument should be fine
detail::VtimesV6<const E1, const E2, const E3, const E4, const E5, const E6> >::type
mul(const E1& e1, const E2& e2, const E3& e3, const E4& e4, const E5& e5, const E6& e6)
{
    return detail::VtimesV6<const E1, const E2, const E3, const E4, const E5, const E6>
    (e1, e2, e3, e4, e5, e6);
}

template<class E1, class E2, class E3, class E4, class E5, class E6, class E7>
typename detail::expression_enabler<E1::is_expression,  // enabler on just the first argument should be fine
detail::VtimesV7<const E1, const E2, const E3, const E4, const E5, const E6, const E7> >::type
mul(const E1& e1, const E2& e2, const E3& e3, const E4& e4, const E5& e5, const E6& e6, const E7& e7)
{
    return detail::VtimesV7<const E1, const E2, const E3, const E4, const E5, const E6, const E7>
    (e1, e2, e3, e4, e5, e6, e7);
}

template<class E1, class E2, class E3, class E4, class E5, class E6, class E7, class E8>
typename detail::expression_enabler<E1::is_expression,  // enabler on just the first argument should be fine
detail::VtimesV8<const E1, const E2, const E3, const E4, const E5, const E6, const E7, const E8> >::type
mul(const E1& e1, const E2& e2, const E3& e3, const E4& e4, const E5& e5, const E6& e6, const E7& e7, const E8& e8)
{
    return detail::VtimesV8<const E1, const E2, const E3, const E4, const E5, const E6, const E7, const E8>
    (e1, e2, e3, e4, e5, e6, e7, e8);
}

template<class E1, class E2, class E3, class E4, class E5, class E6, class E7, class E8, class E9>
typename detail::expression_enabler<E1::is_expression,  // enabler on just the first argument should be fine
detail::VtimesV9<const E1, const E2, const E3, const E4, const E5, const E6, const E7, const E8, const E9> >::type
mul(const E1& e1, const E2& e2, const E3& e3, const E4& e4, const E5& e5, const E6& e6, const E7& e7, const E8& e8, const E9& e9)
{
    return detail::VtimesV9<const E1, const E2, const E3, const E4, const E5, const E6, const E7, const E8, const E9>
    (e1, e2, e3, e4, e5, e6, e7, e8, e9);
}

template<class E1, class E2, class E3, class E4, class E5, class E6, class E7, class E8, class E9, class E10>
typename detail::expression_enabler<E1::is_expression,  // enabler on just the first argument should be fine
detail::VtimesV10<const E1, const E2, const E3, const E4, const E5, const E6, const E7, const E8, const E9, const E10> >::type
mul(const E1& e1, const E2& e2, const E3& e3, const E4& e4, const E5& e5, const E6& e6, const E7& e7, const E8& e8, const E9& e9, const E10& e10)
{
    return detail::VtimesV10<const E1, const E2, const E3, const E4, const E5, const E6, const E7, const E8, const E9, const E10>
    (e1, e2, e3, e4, e5, e6, e7, e8, e9, e10);
}

} // end namespace threx

} // end namespace minlin

#endif
