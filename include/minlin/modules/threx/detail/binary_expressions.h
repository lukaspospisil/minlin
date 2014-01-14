/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_BINARY_EXPRESSIONS_H
#define THREX_DETAIL_BINARY_EXPRESSIONS_H

#include "expression_types.h"
#include "binary_functors.h"
//#include "vector_properties.h"
#include "gemv.h"

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cmath>

namespace minlin {

namespace threx {

namespace detail {

// Scalar-Vector addition
//***********************
template<class Expression>
struct SplusV : public ExpressionType {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<SplusVFunctor<value_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    SplusV(value_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return value + expression[i];
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), SplusVFunctor<value_type>(value));
    }

    const_iterator end() const {
        return const_iterator(expression.end(), SplusVFunctor<value_type>(value));
    }

    difference_type size() const {
        return expression.size();
    }
    
    value_type value;
    const expression_type& expression;
    
};

// Scalar-Vector subtraction
//**************************
template<class Expression>
struct SminusV : public ExpressionType {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<SminusVFunctor<value_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    SminusV(value_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return value - expression[i];
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), SminusVFunctor<value_type>(value));
    }

    const_iterator end() const {
        return const_iterator(expression.end(), SminusVFunctor<value_type>(value));
    }
    
    difference_type size() const {
        return expression.size();
    }
    
    value_type value;
    const expression_type& expression;
    
};

// Scalar-Vector multiplication
//*****************************
template<class Expression>
struct StimesV : public ExpressionType {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<StimesVFunctor<value_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    StimesV(value_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return value * expression[i];
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), StimesVFunctor<value_type>(value));
    }

    const_iterator end() const {
        return const_iterator(expression.end(), StimesVFunctor<value_type>(value));
    }
    
    difference_type size() const {
        return expression.size();
    }
    
    value_type value;
    const expression_type& expression;
    
};

// Scalar-Vector division
//***********************
template<class Expression>
struct SdivideV : public ExpressionType {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<SdivideVFunctor<value_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    SdivideV(value_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return value / expression[i];
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), SdivideVFunctor<value_type>(value));
    }

    const_iterator end() const {
        return const_iterator(expression.end(), SdivideVFunctor<value_type>(value));
    }
    
    difference_type size() const {
        return expression.size();
    }
    
    value_type value;
    const expression_type& expression;
    
};

// Scalar-Vector exponentiation
//*****************************
template<class Expression>
struct SpowerV : public ExpressionType {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<SpowerVFunctor<value_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    SpowerV(value_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        using std::pow;
        return pow(value, expression[i]);
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), SpowerVFunctor<value_type>(value));
    }

    const_iterator end() const {
        return const_iterator(expression.end(), SpowerVFunctor<value_type>(value));
    }
    
    difference_type size() const {
        return expression.size();
    }
    
    value_type value;
    const expression_type& expression;
    
};

// Scalar-Vector arctangent
//*************************
template<class Expression>
struct Satan2V : public ExpressionType {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<Satan2VFunctor<value_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    Satan2V(value_type value, const expression_type& expression) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        using std::atan2;
        return atan2(value, expression[i]);
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), Satan2VFunctor<value_type>(value));
    }

    const_iterator end() const {
        return const_iterator(expression.end(), Satan2VFunctor<value_type>(value));
    }
    
    difference_type size() const {
        return expression.size();
    }
    
    value_type value;
    const expression_type& expression;
    
};

// Vector-Scalar addition
//***********************
template<class Expression>
struct VplusS : public ExpressionType {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VplusSFunctor<value_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VplusS(const expression_type& expression, value_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return expression[i] + value;
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), VplusSFunctor<value_type>(value));
    }

    const_iterator end() const {
        return const_iterator(expression.end(), VplusSFunctor<value_type>(value));
    }
    
    difference_type size() const {
        return expression.size();
    }
    
    value_type value;
    const expression_type& expression;
    
};

// Vector-Scalar subtraction
//**************************
template<class Expression>
struct VminusS : public ExpressionType {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VminusSFunctor<value_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VminusS(const expression_type& expression, value_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return expression[i] - value;
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), VminusSFunctor<value_type>(value));
    }

    const_iterator end() const {
        return const_iterator(expression.end(), VminusSFunctor<value_type>(value));
    }
    
    difference_type size() const {
        return expression.size();
    }
    
    value_type value;
    const expression_type& expression;
    
};

// Vector-Scalar multiplication
//*****************************
template<class Expression>
struct VtimesS : public ExpressionType {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VtimesSFunctor<value_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VtimesS(const expression_type& expression, value_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return expression[i] * value;
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), VtimesSFunctor<value_type>(value));
    }

    const_iterator end() const {
        return const_iterator(expression.end(), VtimesSFunctor<value_type>(value));
    }
    
    difference_type size() const {
        return expression.size();
    }
    
    value_type value;
    const expression_type& expression;
    
};

// Vector-Scalar division
//***********************
template<class Expression>
struct VdivideS : public ExpressionType {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VdivideSFunctor<value_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VdivideS(const expression_type& expression, value_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        return expression[i] / value;
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), VdivideSFunctor<value_type>(value));
    }

    const_iterator end() const {
        return const_iterator(expression.end(), VdivideSFunctor<value_type>(value));
    }
    
    difference_type size() const {
        return expression.size();
    }
    
    value_type value;
    const expression_type& expression;
    
};

// Vector-Scalar exponentiation
//*****************************
template<class Expression>
struct VpowerS : public ExpressionType {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<VpowerSFunctor<value_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    VpowerS(const expression_type& expression, value_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        using std::pow;
        return pow(expression[i], value);
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), VpowerSFunctor<value_type>(value));
    }

    const_iterator end() const {
        return const_iterator(expression.end(), VpowerSFunctor<value_type>(value));
    }
    
    difference_type size() const {
        return expression.size();
    }
    
    value_type value;
    const expression_type& expression;
    
};

// Vector-Scalar arctangent
//*************************
template<class Expression>
struct Vatan2S : public ExpressionType {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::transform_iterator<Vatan2SFunctor<value_type>, typename expression_type::const_iterator> const_iterator;
    typedef const_iterator iterator;

    Vatan2S(const expression_type& expression, value_type value) : value(value), expression(expression) {}

    value_type operator[](difference_type i) const
    {
        using std::atan2;
        return atan2(expression[i], value);
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), Vatan2SFunctor<value_type>(value));
    }

    const_iterator end() const {
        return const_iterator(expression.end(), Vatan2SFunctor<value_type>(value));
    }
    
    difference_type size() const {
        return expression.size();
    }
    
    value_type value;
    const expression_type& expression;
    
};

// Vector-Vector addition
//***********************
template<class LeftExpression, class RightExpression>
struct VplusV : public ExpressionType {

    typedef LeftExpression left_expression;
    typedef RightExpression right_expression;

    typedef typename left_expression::value_type value_type;
    typedef typename left_expression::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::tuple<typename left_expression::const_iterator, typename right_expression::const_iterator> tuple_type;
    typedef ::thrust::zip_iterator<tuple_type> zip_type;
    
    typedef ::thrust::transform_iterator<VplusVFunctor<value_type>, zip_type> const_iterator;
    typedef const_iterator iterator;

    VplusV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        return left[i] + right[i];
    }

    const_iterator begin() const {
        zip_type zip_it(::thrust::make_tuple(left.begin(), right.begin()));
        return const_iterator(zip_it, VplusVFunctor<value_type>());
    }

    const_iterator end() const {
        zip_type zip_it(::thrust::make_tuple(left.end(), right.end()));
        return const_iterator(zip_it, VplusVFunctor<value_type>());
    }

    difference_type size() const {
        return left.size();
    }
    
    const left_expression& left;
    const right_expression& right;
    
};

// Vector-Vector subtraction
//**************************
template<class LeftExpression, class RightExpression>
struct VminusV : public ExpressionType {

    typedef LeftExpression left_expression;
    typedef RightExpression right_expression;

    typedef typename left_expression::value_type value_type;
    typedef typename left_expression::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::tuple<typename left_expression::const_iterator, typename right_expression::const_iterator> tuple_type;
    typedef ::thrust::zip_iterator<tuple_type> zip_type;
    
    typedef ::thrust::transform_iterator<VminusVFunctor<value_type>, zip_type> const_iterator;
    typedef const_iterator iterator;

    VminusV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        return left[i] - right[i];
    }

    const_iterator begin() const {
        zip_type zip_it(::thrust::make_tuple(left.begin(), right.begin()));
        return const_iterator(zip_it, VminusVFunctor<value_type>());
    }

    const_iterator end() const {
        zip_type zip_it(::thrust::make_tuple(left.end(), right.end()));
        return const_iterator(zip_it, VminusVFunctor<value_type>());
    }

    difference_type size() const {
        return left.size();
    }
    
    const left_expression& left;
    const right_expression& right;
    
};

// Vector-Vector multiplication
//*****************************
template<class LeftExpression, class RightExpression>
struct VtimesV : public ExpressionType {

    typedef LeftExpression left_expression;
    typedef RightExpression right_expression;

    typedef typename left_expression::value_type value_type;
    typedef typename left_expression::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::tuple<typename left_expression::const_iterator, typename right_expression::const_iterator> tuple_type;
    typedef ::thrust::transform_iterator<VtimesVFunctor<value_type>, ::thrust::zip_iterator<tuple_type> > const_iterator;
    typedef const_iterator iterator;

    VtimesV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        return left[i] * right[i];
    }

    const_iterator begin() const {
        return const_iterator(::thrust::make_tuple(left.begin(), right.begin()), VtimesVFunctor<value_type>());
    }

    const_iterator end() const {
        return const_iterator(::thrust::make_tuple(left.end(), right.end()), VtimesVFunctor<value_type>());
    }
    
    difference_type size() const {
        return left.size();
    }
    
    const left_expression& left;
    const right_expression& right;
    
};

// Vector-Vector division
//***********************
template<class LeftExpression, class RightExpression>
struct VdivideV : public ExpressionType {

    typedef LeftExpression left_expression;
    typedef RightExpression right_expression;

    typedef typename left_expression::value_type value_type;
    typedef typename left_expression::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::tuple<typename left_expression::const_iterator, typename right_expression::const_iterator> tuple_type;
    typedef ::thrust::transform_iterator<VdivideVFunctor<value_type>, ::thrust::zip_iterator<tuple_type> > const_iterator;
    typedef const_iterator iterator;

    VdivideV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        return left[i] / right[i];
    }

    const_iterator begin() const {
        return const_iterator(::thrust::make_tuple(left.begin(), right.begin()), VdivideVFunctor<value_type>());
    }

    const_iterator end() const {
        return const_iterator(::thrust::make_tuple(left.end(), right.end()), VdivideVFunctor<value_type>());
    }
    
    difference_type size() const {
        return left.size();
    }
    
    const left_expression& left;
    const right_expression& right;
    
};

// Vector-Vector exponentiation
//*****************************
template<class LeftExpression, class RightExpression>
struct VpowerV : public ExpressionType {

    typedef LeftExpression left_expression;
    typedef RightExpression right_expression;

    typedef typename left_expression::value_type value_type;
    typedef typename left_expression::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::tuple<typename left_expression::const_iterator, typename right_expression::const_iterator> tuple_type;
    typedef ::thrust::transform_iterator<VpowerVFunctor<value_type>, ::thrust::zip_iterator<tuple_type> > const_iterator;
    typedef const_iterator iterator;

    VpowerV(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        using std::pow;
        return pow(left[i], right[i]);
    }

    const_iterator begin() const {
        return const_iterator(::thrust::make_tuple(left.begin(), right.begin()), VpowerVFunctor<value_type>());
    }

    const_iterator end() const {
        return const_iterator(::thrust::make_tuple(left.end(), right.end()), VpowerVFunctor<value_type>());
    }

    difference_type size() const {
        return left.size();
    }

    const left_expression& left;
    const right_expression& right;

};

// Vector-Vector arctangent
//*************************
template<class LeftExpression, class RightExpression>
struct Vatan2V : public ExpressionType {

    typedef LeftExpression left_expression;
    typedef RightExpression right_expression;

    typedef typename left_expression::value_type value_type;
    typedef typename left_expression::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    typedef ::thrust::tuple<typename left_expression::const_iterator, typename right_expression::const_iterator> tuple_type;
    typedef ::thrust::transform_iterator<Vatan2VFunctor<value_type>, ::thrust::zip_iterator<tuple_type> > const_iterator;
    typedef const_iterator iterator;

    Vatan2V(const left_expression& left, const right_expression& right) : left(left), right(right) {}

    value_type operator[](difference_type i) const
    {
        using std::atan2;
        return atan2(left[i], right[i]);
    }

    const_iterator begin() const {
        return const_iterator(::thrust::make_tuple(left.begin(), right.begin()), Vatan2VFunctor<value_type>());
    }

    const_iterator end() const {
        return const_iterator(::thrust::make_tuple(left.end(), right.end()), Vatan2VFunctor<value_type>());
    }
    
    difference_type size() const {
        return left.size();
    }
    
    const left_expression& left;
    const right_expression& right;
    
};

// Matrix-Vector multiplication
//***********************
template<class LeftExpression, class RightExpression>
struct MtimesV : public ExpressionType {

    // tag as a level-2 operator
    typedef int is_specialized;

    typedef LeftExpression left_expression;
    typedef RightExpression right_expression;

    typedef typename left_expression::value_type value_type;
    typedef typename left_expression::difference_type difference_type;
    struct reference{};
    typedef reference const_reference;
    struct pointer{};
    typedef pointer const_pointer;

    // no iterator types
    struct const_iterator{};
    typedef const_iterator iterator;

    MtimesV(const left_expression& left,
            const right_expression& right,
            difference_type rows,
            difference_type cols,
            value_type alpha=value_type(1))
    :   left(left),
        right(right),
        rows(rows),
        cols(cols),
        alpha(alpha)
    {}

    // no access operator[] or begin()/end()
    difference_type size() const {
        return left.size()/right.size();
    }

    // apply matrix vector multiplication
    template <typename LValue>
    void apply(LValue &lhs, value_type beta=value_type(0)) const {
        value_type *y = thrust::raw_pointer_cast( lhs.data() );
        value_type const *A = thrust::raw_pointer_cast( left.data() );
        value_type const *x = thrust::raw_pointer_cast( right.data() );

        // how to get data out?
        // option 1:
        //value_type const *x = vector_properties<right_expression>::pointer(right);
        //int incx            = vector_properties<right_expression>::stride(right);
        //int lenx            = vector_properties<right_expression>::size(right);

        // option 2:
        // make both the vector and range_wrapper types provide the same interface
        // the problem is that stride() is not a sensible property for a vector to carry
        //value_type const *x = right.pointer();
        //int incx            = right.stride();
        //int lenx            = right.size();

        // we want to extract stride information
        int  incx=1;
        int  incy=1;
        int  lda=rows;
        char trans='N';

        /////////////////////////////////////////////////////////////////////////////////////////////////////
        std::cout << "apply::lhs has dimension " << lhs.size() << std::endl;
        std::cout << "apply::lhs has pointer " << y << std::endl;
        std::cout << "apply::rows, cols and alpha " << rows << ", " << cols << ", " << alpha << std::endl;
        std::cout << "value_type == " << print_traits<value_type>::print() << std::endl;
        /////////////////////////////////////////////////////////////////////////////////////////////////////

        gemv_host(A, x, y, alpha, beta, int(rows), int(cols), int(incx), int(incy), int(lda), trans);
    }

    const left_expression& left;
    const right_expression& right;
    const value_type alpha;
    const difference_type rows;
    const difference_type cols;
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
