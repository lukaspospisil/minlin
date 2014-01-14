#pragma once

#include "storage.h"
#include "indexing_expressions.h"

// interface for querying expressions for properties used in specialized operations
//
// Ben Cumming

namespace minlin {
namespace threx {
namespace detail {

template <
    typename Expression,
    typename ExpressionCondition= typename expression_enabler<Expression::is_expression, void>::type >
struct vector_properties {
    typedef typename Expression::value_type value_type;
    typedef typename Expression::difference_type difference_type;

    // the size of the expression
    static difference_type size(Expression const& expression) {
        return expression.size();
    }

    // default expression has stride 1
    static difference_type stride(Expression const& expression) {
        return difference_type(1);
    }

    // expressions that are not in place don't have pointers
    //static value_type* pointer(Expression const& expression) {}
};

// specializations:
//      raw underlying data (ByValue<>)
//      accessed via a unit stride range
//      accessed via a non-unit stride range
//      accessed via an index vector
//          - not handled yet, but may be required for matrix operators when doing ops like
//              matrix A = B(all, p);
//            where p is an index vector that specifies a set of columns to select

// specialization for non unit stride indexes
template <typename Expression>
struct vector_properties<
    class VindexRangeNonunitStride<Expression>,
    void>
{
    typedef typename Expression::value_type value_type;
    typedef typename Expression::difference_type difference_type;

    typedef class VindexRangeNonunitStride<Expression> IndexedExpression;
                     //VindexRangeNonunitStride

    // the size of the expression
    difference_type size(IndexedExpression const& expression) {
        return expression.size();
    }

    // unit stride index can hard coded it to 1
    difference_type stride(IndexedExpression const& expression) {
        return difference_type(1);
    }

    // pointer to data
    value_type* pointer(IndexedExpression const& expression) {
        // VindexRangeNonunitStride is derived derived from InPlace, so it's expression
        // will provide access to underlying data
        return (value_type*)(thrust::raw_pointer_cast( expression.expression.data() ));
    }
};

// specialization for non unit stride indexes
template <typename Expression>
struct vector_properties<class VindexRangeUnitStride<Expression>, void>
{
    typedef typename Expression::value_type value_type;
    typedef typename Expression::difference_type difference_type;

    typedef class VindexRangeUnitStride<Expression> IndexedExpression;

    // the size of the expression
    difference_type size(IndexedExpression const& expression) {
        return expression.size();
    }

    // default expression has stride 1
    difference_type size(IndexedExpression const& expression) {
        return expression.stride();
    }

    // pointer to data
    value_type* pointer(IndexedExpression const& expression) {
        // VindexRangeUnitStride is derived derived from InPlace, so it's expression
        // will provide access to underlying data
        return (value_type*)(thrust::raw_pointer_cast( expression.expression.data() ));
    }
};

// specialization for ByValue (raw data)
template <typename Data>
struct vector_properties<class ByValue<Data>, void>
{
    typedef typename Expression::value_type value_type;
    typedef typename Expression::difference_type difference_type;

    typedef class ByValue<Data> Expression;

    // the size of the expression
    difference_type size(Expression const& expression) {
        return expression.size();
    }

    // default expression has stride 1
    difference_type size(Expression const& expression) {
        return expression.stride();
    }

    // pointer to data
    value_type* pointer(Expression const& expression) {
        // ByValue expression provides pointer to data
        return (value_type*)(thrust::raw_pointer_cast( expression.data() ));
    }
};

} // end namespace detail
} // end namespace threx
} // end namespace minlin

