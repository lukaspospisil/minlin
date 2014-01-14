/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_EXPRESSION_TYPES_H
#define THREX_EXPRESSION_TYPES_H

namespace minlin {

namespace threx {

namespace detail {

// Declare all the expression types supported by threx
template<class Expression> class absV;
template<class Expression> class acosV;
template<class Expression> class asinV;
template<class Expression> class atanV;
template<class Expression> class ceilV;
template<class Expression> class cosV;
template<class Expression> class coshV;
template<class Expression> class expV;
template<class Expression> class floorV;
template<class Expression> class logV;
template<class Expression> class log10V;
template<class Expression> class sinV;
template<class Expression> class sinhV;
template<class Expression> class sqrtV;
template<class Expression> class tanV;
template<class Expression> class tanhV;

template<class Expression> class plusV;
template<class Expression> class minusV;

template<class Expression> class SplusV;
template<class Expression> class SminusV;
template<class Expression> class StimesV;
template<class Expression> class SdivideV;
template<class Expression> class SpowerV;
template<class Expression> class Satan2V;

template<class Expression> class SequalToV;
template<class Expression> class SnotEqualToV;
template<class Expression> class SlessThanV;
template<class Expression> class SlessThanOrEqualToV;
template<class Expression> class SgreaterThanV;
template<class Expression> class SgreaterThanOrEqualToV;

template<class Expression> class SorV;
template<class Expression> class SandV;

template<class Expression> class VplusS;
template<class Expression> class VminusS;
template<class Expression> class VtimesS;
template<class Expression> class VdivideS;
template<class Expression> class VpowerS;
template<class Expression> class Vatan2S;

template<class Expression> class VequalToS;
template<class Expression> class VnotEqualToS;
template<class Expression> class VlessThanS;
template<class Expression> class VlessThanOrEqualToS;
template<class Expression> class VgreaterThanS;
template<class Expression> class VgreaterThanOrEqualToS;

template<class Expression> class VorS;
template<class Expression> class VandS;

template<class LeftExpression, class RightExpression> class VplusV;
template<class LeftExpression, class RightExpression> class VminusV;
template<class LeftExpression, class RightExpression> class VtimesV;
template<class LeftExpression, class RightExpression> class VdivideV;
template<class LeftExpression, class RightExpression> class VpowerV;
template<class LeftExpression, class RightExpression> class Vatan2V;

template<class LeftExpression, class RightExpression> class VequalToV;
template<class LeftExpression, class RightExpression> class VnotEqualToV;
template<class LeftExpression, class RightExpression> class VlessThanV;
template<class LeftExpression, class RightExpression> class VlessThanOrEqualToV;
template<class LeftExpression, class RightExpression> class VgreaterThanV;
template<class LeftExpression, class RightExpression> class VgreaterThanOrEqualToV;

template<class LeftExpression, class RightExpression> class VorV;
template<class LeftExpression, class RightExpression> class VandV;
template<class Expression> class notV;

template<class E1, class E2, class E3> class VtimesV3;
template<class E1, class E2, class E3, class E4> class VtimesV4;
template<class E1, class E2, class E3, class E4, class E5> class VtimesV5;
template<class E1, class E2, class E3, class E4, class E5,
         class E6> class VtimesV6;
template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7> class VtimesV7;
template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7, class E8> class VtimesV8;
template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7, class E8, class E9> class VtimesV9;
template<class E1, class E2, class E3, class E4, class E5,
         class E6, class E7, class E8, class E9, class E10> class VtimesV10;

template<class Expression> class InPlace;
template<class Expression> class VindexRangeUnitStride;
template<class Expression> class VindexRangeNonunitStride;
template<class Expression, class IndexExpression> class VindexI;

template<class Expresssion> class VdoubleIndexRangeUnitStride;

/****************************
 * BLAS level two operators *
 ****************************/
template<class Left, class Right> class MtimesV;

// Map these types to the agreed-upon names so that minlin types can find them
struct ExpressionType {

    // Used to selectively enable operator functions for expresion types only
    enum {is_expression};

    // Vector unary operators
    // **********************
/*
    template<class Expression>
    struct cos_v {
        typedef cosV<const Expression> type;
    };
*/

    #define THREX_DEFINE_UNARY_OPERATOR_TYPE(op) \
    template<class Expression> \
    struct op##_v { \
        typedef op##V<const Expression> type; \
    };
    
THREX_DEFINE_UNARY_OPERATOR_TYPE(abs)
THREX_DEFINE_UNARY_OPERATOR_TYPE(acos)
THREX_DEFINE_UNARY_OPERATOR_TYPE(asin)
THREX_DEFINE_UNARY_OPERATOR_TYPE(atan)
THREX_DEFINE_UNARY_OPERATOR_TYPE(ceil)
THREX_DEFINE_UNARY_OPERATOR_TYPE(cos)
THREX_DEFINE_UNARY_OPERATOR_TYPE(cosh)
THREX_DEFINE_UNARY_OPERATOR_TYPE(exp)
THREX_DEFINE_UNARY_OPERATOR_TYPE(floor)
THREX_DEFINE_UNARY_OPERATOR_TYPE(log)
THREX_DEFINE_UNARY_OPERATOR_TYPE(log10)
THREX_DEFINE_UNARY_OPERATOR_TYPE(sin)
THREX_DEFINE_UNARY_OPERATOR_TYPE(sinh)
THREX_DEFINE_UNARY_OPERATOR_TYPE(sqrt)
THREX_DEFINE_UNARY_OPERATOR_TYPE(tan)
THREX_DEFINE_UNARY_OPERATOR_TYPE(tanh)

    template<class Expression>
    struct plus_v {
        typedef plusV<const Expression> type;
    };

    template<class Expression>
    struct minus_v {
        typedef minusV<const Expression> type;
    };

    // Scalar-Vector operators
    // ***********************
    template<class Expression>
    struct s_plus_v {
        typedef SplusV<const Expression> type;
    };

    template<class Expression>
    struct s_minus_v {
        typedef SminusV<const Expression> type;
    };

    template<class Expression>
    struct s_times_v {
        typedef StimesV<const Expression> type;
    };

    template<class Expression>
    struct s_divide_v {
        typedef SdivideV<const Expression> type;
    };

    template<class Expression>
    struct s_power_v {
        typedef SpowerV<const Expression> type;
    };

    template<class Expression>
    struct s_atan2_v {
        typedef Satan2V<const Expression> type;
    };

    template<class Expression>
    struct s_equal_to_v {
        typedef SequalToV<const Expression> type;
    };

    template<class Expression>
    struct s_not_equal_to_v {
        typedef SnotEqualToV<const Expression> type;
    };

    template<class Expression>
    struct s_less_than_or_equal_to_v {
        typedef SlessThanOrEqualToV<const Expression> type;
    };

    template<class Expression>
    struct s_less_than_v {
        typedef SlessThanV<const Expression> type;
    };

    template<class Expression>
    struct s_greater_than_v {
        typedef SgreaterThanV<const Expression> type;
    };

    template<class Expression>
    struct s_greater_than_or_equal_to_v {
        typedef SgreaterThanOrEqualToV<const Expression> type;
    };

    template<class Expression>
    struct s_or_v {
        typedef SorV<const Expression> type;
    };

    template<class Expression>
    struct s_and_v {
        typedef SandV<const Expression> type;
    };

    // Vector-Scalar operators
    // ***********************
    template<class Expression>
    struct v_plus_s {
        typedef VplusS<const Expression> type;
    };

    template<class Expression>
    struct v_minus_s {
        typedef VminusS<const Expression> type;
    };

    template<class Expression>
    struct v_times_s {
        typedef VtimesS<const Expression> type;
    };

    template<class Expression>
    struct v_divide_s {
        typedef VdivideS<const Expression> type;
    };

    template<class Expression>
    struct v_power_s {
        typedef VpowerS<const Expression> type;
    };

    template<class Expression>
    struct v_atan2_s {
        typedef Vatan2S<const Expression> type;
    };

    template<class Expression>
    struct v_equal_to_s {
        typedef VequalToS<const Expression> type;
    };

    template<class Expression>
    struct v_not_equal_to_s {
        typedef VnotEqualToS<const Expression> type;
    };

    template<class Expression>
    struct v_less_than_s {
        typedef VlessThanS<const Expression> type;
    };

    template<class Expression>
    struct v_less_than_or_equal_to_s {
        typedef VlessThanOrEqualToS<const Expression> type;
    };

    template<class Expression>
    struct v_greater_than_s {
        typedef VgreaterThanS<const Expression> type;
    };

    template<class Expression>
    struct v_greater_than_or_equal_to_s {
        typedef VgreaterThanOrEqualToS<const Expression> type;
    };

    template<class Expression>
    struct v_or_s {
        typedef VorS<const Expression> type;
    };

    template<class Expression>
    struct v_and_s {
        typedef VandS<const Expression> type;
    };

    // Vector-Vector operators
    // ***********************
    template<class LeftExpression, class RightExpression>
    struct v_plus_v {
        typedef VplusV<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_minus_v {
        typedef VminusV<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_times_v {
        typedef VtimesV<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_divide_v {
        typedef VdivideV<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_power_v {
        typedef VpowerV<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_atan2_v {
        typedef Vatan2V<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_equal_to_v {
        typedef VequalToV<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_not_equal_to_v {
        typedef VnotEqualToV<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_less_than_v {
        typedef VlessThanV<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_less_than_or_equal_to_v {
        typedef VlessThanOrEqualToV<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_greater_than_v {
        typedef VgreaterThanV<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_greater_than_or_equal_to_v {
        typedef VgreaterThanOrEqualToV<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_or_v {
        typedef VorV<const LeftExpression, const RightExpression> type;
    };

    template<class LeftExpression, class RightExpression>
    struct v_and_v {
        typedef VandV<const LeftExpression, const RightExpression> type;
    };

    template<class Expression>
    struct not_v {
        typedef notV<const Expression> type;
    };

    // Vector-Vector-...-Vector operators
    // **********************************
    template<class E1, class E2, class E3>
    struct v_times_v3 {
        typedef VtimesV3<const E1, const E2, const E3> type;
    };

    template<class E1, class E2, class E3, class E4>
    struct v_times_v4 {
        typedef VtimesV4<const E1, const E2, const E3, const E4> type;
    };

    template<class E1, class E2, class E3, class E4, class E5>
    struct v_times_v5 {
        typedef VtimesV5<const E1, const E2, const E3, const E4, const E5> type;
    };

    template<class E1, class E2, class E3, class E4, class E5,
             class E6>
    struct v_times_v6 {
        typedef VtimesV6<const E1, const E2, const E3, const E4, const E5,
                         const E6> type;
    };

    template<class E1, class E2, class E3, class E4, class E5,
             class E6, class E7>
    struct v_times_v7 {
        typedef VtimesV7<const E1, const E2, const E3, const E4, const E5,
                         const E6, const E7> type;
    };

    template<class E1, class E2, class E3, class E4, class E5,
             class E6, class E7, class E8>
    struct v_times_v8 {
        typedef VtimesV8<const E1, const E2, const E3, const E4, const E5,
                         const E6, const E7, const E8> type;
    };

    template<class E1, class E2, class E3, class E4, class E5,
             class E6, class E7, class E8, class E9>
    struct v_times_v9 {
        typedef VtimesV9<const E1, const E2, const E3, const E4, const E5,
                         const E6, const E7, const E8, const E9> type;
    };

    template<class E1, class E2, class E3, class E4, class E5,
             class E6, class E7, class E8, class E9, class E10>
    struct v_times_v10 {
        typedef VtimesV10<const E1, const E2, const E3, const E4, const E5,
                          const E6, const E7, const E8, const E9, const E10> type;
    };

    // Most types that inherit from this base do not have in-place assignment
    // semantics.  Those that do override in_place to be true.
    enum {is_inplace = false};

    // In-place indexing
    template<class Expression>
    struct inplace {
        typedef InPlace<Expression> type;
    };

    // Range indexing
    template<class Expression>
    struct index_range_unit_stride {
        typedef VindexRangeUnitStride<Expression> type;
    };

    template<class Expression>
    struct index_range_nonunit_stride {
        typedef VindexRangeNonunitStride<Expression> type;
    };

    template<class Expression>
    struct double_index_range_unit_stride {
        typedef VdoubleIndexRangeUnitStride<Expression> type;
    };

    // Indirect vector indexing
    template<class Expression, class IndexExpression>
    struct indirect_index {
        typedef VindexI<Expression, const IndexExpression> type;
    };

    // Matrix-Vector operators
    // matrix-vector multiplication
    template<class Mat, class Vec>
    struct m_times_v {
        typedef MtimesV<Mat, Vec> type;
    };
};

// Simple enabler/disabler for expression types
template<int Ignore, typename ReturnType>
struct expression_enabler {
    typedef ReturnType type;
};

template<int Ignore1, int Ignore2, typename ReturnType>
struct expressions_enabler {
    typedef ReturnType type;
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
