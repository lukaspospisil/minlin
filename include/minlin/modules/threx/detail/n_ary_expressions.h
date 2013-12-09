/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_N_ARY_EXPRESSIONS_H
#define THREX_DETAIL_N_ARY_EXPRESSIONS_H

#include "expression_types.h"
#include "n_ary_functors.h"

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace minlin {

namespace threx {

namespace detail {

// Vector-Vector multiplication
//*****************************
template<class E1, class E2, class E3>
struct VtimesV3 : public ExpressionType {

	typedef typename E1::value_type value_type;
    typedef typename E1::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

	typedef ::thrust::tuple<
	    typename E1::const_iterator,
	    typename E2::const_iterator,
	    typename E3::const_iterator
    > tuple_type;
    typedef ::thrust::transform_iterator<VtimesVFunctor3<value_type>, ::thrust::zip_iterator<tuple_type> > const_iterator;
    typedef const_iterator iterator;

    VtimesV3(const E1& e1, const E2& e2, const E3& e3)
        : e1(e1), e2(e2), e3(e3) {}

    value_type operator[](difference_type i) const
    {
        return e1[i] * e2[i] * e3[i];
    }

	const_iterator begin() const {
		return const_iterator(::thrust::make_tuple(
		    e1.begin(), e2.begin(), e3.begin()),
		    VtimesVFunctor3<value_type>());
	}

	const_iterator end() const {
		return const_iterator(::thrust::make_tuple(
		    e1.end(), e2.end(), e3.end()),
            VtimesVFunctor3<value_type>());
	}
	
	difference_type size() const {
		return e1.size();
	}
	
	const E1& e1;
	const E2& e2;
	const E3& e3;
	
};

template<class E1, class E2, class E3, class E4>
struct VtimesV4 : public ExpressionType {

	typedef typename E1::value_type value_type;
    typedef typename E1::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

	typedef ::thrust::tuple<
	    typename E1::const_iterator,
	    typename E2::const_iterator,
	    typename E3::const_iterator,
	    typename E4::const_iterator
    > tuple_type;
    typedef ::thrust::transform_iterator<VtimesVFunctor4<value_type>, ::thrust::zip_iterator<tuple_type> > const_iterator;
    typedef const_iterator iterator;

    VtimesV4(const E1& e1, const E2& e2, const E3& e3, const E4& e4)
        : e1(e1), e2(e2), e3(e3), e4(e4) {}

    value_type operator[](difference_type i) const
    {
        return e1[i] * e2[i] * e3[i] * e4[i];
    }

	const_iterator begin() const {
		return const_iterator(::thrust::make_tuple(
		    e1.begin(), e2.begin(), e3.begin(), e4.begin()),
		    VtimesVFunctor4<value_type>());
	}

	const_iterator end() const {
		return const_iterator(::thrust::make_tuple(
		    e1.end(), e2.end(), e3.end(), e4.end()),
            VtimesVFunctor4<value_type>());
	}
	
	difference_type size() const {
		return e1.size();
	}
	
	const E1& e1;
	const E2& e2;
	const E3& e3;
	const E4& e4;
	
};

template<class E1, class E2, class E3, class E4, class E5>
struct VtimesV5 : public ExpressionType {

	typedef typename E1::value_type value_type;
    typedef typename E1::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

	typedef ::thrust::tuple<
	    typename E1::const_iterator,
	    typename E2::const_iterator,
	    typename E3::const_iterator,
	    typename E4::const_iterator,
	    typename E5::const_iterator
    > tuple_type;
    typedef ::thrust::transform_iterator<VtimesVFunctor5<value_type>, ::thrust::zip_iterator<tuple_type> > const_iterator;
    typedef const_iterator iterator;

    VtimesV5(const E1& e1, const E2& e2, const E3& e3, const E4& e4, const E5& e5)
        : e1(e1), e2(e2), e3(e3), e4(e4), e5(e5) {}

    value_type operator[](difference_type i) const
    {
        return e1[i] * e2[i] * e3[i] * e4[i] * e5[i];
    }

	const_iterator begin() const {
		return const_iterator(::thrust::make_tuple(
		    e1.begin(), e2.begin(), e3.begin(), e4.begin(), e5.begin()),
		    VtimesVFunctor5<value_type>());
	}

	const_iterator end() const {
		return const_iterator(::thrust::make_tuple(
		    e1.end(), e2.end(), e3.end(), e4.end(), e5.end()),
            VtimesVFunctor5<value_type>());
	}
	
	difference_type size() const {
		return e1.size();
	}
	
	const E1& e1;
	const E2& e2;
	const E3& e3;
	const E4& e4;
	const E5& e5;
	
};

template<class E1, class E2, class E3, class E4, class E5,
         class E6>
struct VtimesV6 : public ExpressionType {

	typedef typename E1::value_type value_type;
    typedef typename E1::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

	typedef ::thrust::tuple<
	    typename E1::const_iterator,
	    typename E2::const_iterator,
	    typename E3::const_iterator,
	    typename E4::const_iterator,
	    typename E5::const_iterator,
        typename E6::const_iterator
    > tuple_type;
    typedef ::thrust::transform_iterator<VtimesVFunctor6<value_type>, ::thrust::zip_iterator<tuple_type> > const_iterator;
    typedef const_iterator iterator;

    VtimesV6(const E1& e1, const E2& e2, const E3& e3, const E4& e4, const E5& e5,
             const E6& e6)
        : e1(e1), e2(e2), e3(e3), e4(e4), e5(e5),
          e6(e6) {}

    value_type operator[](difference_type i) const
    {
        return e1[i] * e2[i] * e3[i] * e4[i] * e5[i]
             * e6[i];
    }

	const_iterator begin() const {
		return const_iterator(::thrust::make_tuple(
		    e1.begin(), e2.begin(), e3.begin(), e4.begin(), e5.begin(),
		    e6.begin()),
		    VtimesVFunctor6<value_type>());
	}

	const_iterator end() const {
		return const_iterator(::thrust::make_tuple(
		    e1.end(), e2.end(), e3.end(), e4.end(), e5.end(),
		    e6.end()),
            VtimesVFunctor6<value_type>());
	}
	
	difference_type size() const {
		return e1.size();
	}
	
	const E1& e1;
	const E2& e2;
	const E3& e3;
	const E4& e4;
	const E5& e5;
    const E6& e6;
	
};

template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7>
struct VtimesV7 : public ExpressionType {

	typedef typename E1::value_type value_type;
    typedef typename E1::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

	typedef ::thrust::tuple<
	    typename E1::const_iterator,
	    typename E2::const_iterator,
	    typename E3::const_iterator,
	    typename E4::const_iterator,
	    typename E5::const_iterator,
        typename E6::const_iterator,
        typename E7::const_iterator
    > tuple_type;
    typedef ::thrust::transform_iterator<VtimesVFunctor7<value_type>, ::thrust::zip_iterator<tuple_type> > const_iterator;
    typedef const_iterator iterator;

    VtimesV7(const E1& e1, const E2& e2, const E3& e3, const E4& e4, const E5& e5,
             const E6& e6, const E7& e7)
        : e1(e1), e2(e2), e3(e3), e4(e4), e5(e5),
          e6(e6), e7(e7) {}

    value_type operator[](difference_type i) const
    {
        return e1[i] * e2[i] * e3[i] * e4[i] * e5[i]
             * e6[i] * e7[i];
    }

	const_iterator begin() const {
		return const_iterator(::thrust::make_tuple(
		    e1.begin(), e2.begin(), e3.begin(), e4.begin(), e5.begin(),
		    e6.begin(), e7.begin()),
		    VtimesVFunctor7<value_type>());
	}

	const_iterator end() const {
		return const_iterator(::thrust::make_tuple(
		    e1.end(), e2.end(), e3.end(), e4.end(), e5.end(),
		    e6.end(), e7.end()),
            VtimesVFunctor7<value_type>());
	}
	
	difference_type size() const {
		return e1.size();
	}
	
	const E1& e1;
	const E2& e2;
	const E3& e3;
	const E4& e4;
	const E5& e5;
    const E6& e6;
    const E7& e7;
	
};

template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7, class E8>
struct VtimesV8 : public ExpressionType {

	typedef typename E1::value_type value_type;
    typedef typename E1::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

	typedef ::thrust::tuple<
	    typename E1::const_iterator,
	    typename E2::const_iterator,
	    typename E3::const_iterator,
	    typename E4::const_iterator,
	    typename E5::const_iterator,
        typename E6::const_iterator,
        typename E7::const_iterator,
        typename E8::const_iterator
    > tuple_type;
    typedef ::thrust::transform_iterator<VtimesVFunctor8<value_type>, ::thrust::zip_iterator<tuple_type> > const_iterator;
    typedef const_iterator iterator;

    VtimesV8(const E1& e1, const E2& e2, const E3& e3, const E4& e4, const E5& e5,
             const E6& e6, const E7& e7, const E8& e8)
        : e1(e1), e2(e2), e3(e3), e4(e4), e5(e5),
          e6(e6), e7(e7), e8(e8) {}

    value_type operator[](difference_type i) const
    {
        return e1[i] * e2[i] * e3[i] * e4[i] * e5[i]
             * e6[i] * e7[i] * e8[i];
    }

	const_iterator begin() const {
		return const_iterator(::thrust::make_tuple(
		    e1.begin(), e2.begin(), e3.begin(), e4.begin(), e5.begin(),
		    e6.begin(), e7.begin(), e8.begin()),
		    VtimesVFunctor8<value_type>());
	}

	const_iterator end() const {
		return const_iterator(::thrust::make_tuple(
		    e1.end(), e2.end(), e3.end(), e4.end(), e5.end(),
		    e6.end(), e7.end(), e8.end()),
            VtimesVFunctor8<value_type>());
	}
	
	difference_type size() const {
		return e1.size();
	}
	
	const E1& e1;
	const E2& e2;
	const E3& e3;
	const E4& e4;
	const E5& e5;
    const E6& e6;
    const E7& e7;
    const E8& e8;
	
};

template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7, class E8, class E9>
struct VtimesV9 : public ExpressionType {

	typedef typename E1::value_type value_type;
    typedef typename E1::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

	typedef ::thrust::tuple<
	    typename E1::const_iterator,
	    typename E2::const_iterator,
	    typename E3::const_iterator,
	    typename E4::const_iterator,
	    typename E5::const_iterator,
        typename E6::const_iterator,
        typename E7::const_iterator,
        typename E8::const_iterator,
        typename E9::const_iterator
    > tuple_type;
    typedef ::thrust::transform_iterator<VtimesVFunctor9<value_type>, ::thrust::zip_iterator<tuple_type> > const_iterator;
    typedef const_iterator iterator;

    VtimesV9(const E1& e1, const E2& e2, const E3& e3, const E4& e4, const E5& e5,
             const E6& e6, const E7& e7, const E8& e8, const E9& e9)
        : e1(e1), e2(e2), e3(e3), e4(e4), e5(e5),
          e6(e6), e7(e7), e8(e8), e9(e9) {}

    value_type operator[](difference_type i) const
    {
        return e1[i] * e2[i] * e3[i] * e4[i] * e5[i]
             * e6[i] * e7[i] * e8[i] * e9[i];
    }

	const_iterator begin() const {
		return const_iterator(::thrust::make_tuple(
		    e1.begin(), e2.begin(), e3.begin(), e4.begin(), e5.begin(),
		    e6.begin(), e7.begin(), e8.begin(), e9.begin()),
		    VtimesVFunctor9<value_type>());
	}

	const_iterator end() const {
		return const_iterator(::thrust::make_tuple(
		    e1.end(), e2.end(), e3.end(), e4.end(), e5.end(),
		    e6.end(), e7.end(), e8.end(), e9.end()),
            VtimesVFunctor9<value_type>());
	}
	
	difference_type size() const {
		return e1.size();
	}
	
	const E1& e1;
	const E2& e2;
	const E3& e3;
	const E4& e4;
	const E5& e5;
    const E6& e6;
    const E7& e7;
    const E8& e8;
    const E9& e9;
	
};

template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7, class E8, class E9, class E10>
struct VtimesV10 : public ExpressionType {

	typedef typename E1::value_type value_type;
    typedef typename E1::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

	typedef ::thrust::tuple<
	    typename E1::const_iterator,
	    typename E2::const_iterator,
	    typename E3::const_iterator,
	    typename E4::const_iterator,
	    typename E5::const_iterator,
        typename E6::const_iterator,
        typename E7::const_iterator,
        typename E8::const_iterator,
        typename E9::const_iterator,
        typename E10::const_iterator
    > tuple_type;
    typedef ::thrust::transform_iterator<VtimesVFunctor10<value_type>, ::thrust::zip_iterator<tuple_type> > const_iterator;
    typedef const_iterator iterator;

    VtimesV10(const E1& e1, const E2& e2, const E3& e3, const E4& e4, const E5& e5,
              const E6& e6, const E7& e7, const E8& e8, const E9& e9, const E10& e10)
        : e1(e1), e2(e2), e3(e3), e4(e4), e5(e5),
          e6(e6), e7(e7), e8(e8), e9(e9), e10(e10) {}

    value_type operator[](difference_type i) const
    {
        return e1[i] * e2[i] * e3[i] * e4[i] * e5[i]
             * e6[i] * e7[i] * e8[i] * e9[i] * e10[i];
    }

	const_iterator begin() const {
		return const_iterator(::thrust::make_tuple(
		    e1.begin(), e2.begin(), e3.begin(), e4.begin(), e5.begin(),
		    e6.begin(), e7.begin(), e8.begin(), e9.begin(), e10.begin()),
		    VtimesVFunctor10<value_type>());
	}

	const_iterator end() const {
		return const_iterator(::thrust::make_tuple(
		    e1.end(), e2.end(), e3.end(), e4.end(), e5.end(),
		    e6.end(), e7.end(), e8.end(), e9.end(), e10.end()),
            VtimesVFunctor10<value_type>());
	}
	
	difference_type size() const {
		return e1.size();
	}
	
	const E1& e1;
	const E2& e2;
	const E3& e3;
	const E4& e4;
	const E5& e5;
    const E6& e6;
    const E7& e7;
    const E8& e8;
    const E9& e9;
    const E10& e10;

};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
