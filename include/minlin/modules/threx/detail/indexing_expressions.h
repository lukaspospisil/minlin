/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_INDEXING_EXPRESSIONS_H
#define THREX_DETAIL_INDEXING_EXPRESSIONS_H

#include "assignment.h"
#include "inplace.h"
#include "function_functors.h"
#include "indexing_functors.h"

#include <thrust/iterator/transform_iterator.h>

#include <cmath>

namespace minlin {

namespace threx {
    
namespace detail {

template<class Expression>
struct VindexRangeUnitStride :
    public InPlaceOps<VindexRangeUnitStride<Expression>, typename Expression::value_type> {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    typedef typename expression_type::reference reference;
    typedef typename expression_type::const_reference const_reference;
    typedef typename expression_type::pointer pointer;
    typedef typename expression_type::const_pointer const_pointer;
    typedef typename expression_type::iterator iterator;
    typedef typename expression_type::const_iterator const_iterator;

    typedef InPlaceOps<VindexRangeUnitStride, value_type> base;

    VindexRangeUnitStride(expression_type& expression,
            difference_type start, difference_type finish)
        : expression(expression),
          start(start), dim(std::max(difference_type(), finish - start + 1))
    {}

    // Bring in the base assignment operator templates
    using base::operator=;

    // The compiler won't synthesise an assignment operator, since we 
    // have a reference member.  We want something different anyway.
    VindexRangeUnitStride& operator=(const VindexRangeUnitStride& other)
    {
        assign_in_place(*this, other);
        return *this;
    }

    reference operator[](difference_type i)
    {
        return expression[i + start];
    }

    value_type operator[](difference_type i) const
    {
        return expression[i + start];
    }

    iterator begin() {
        return expression.begin() + start;
    }

    iterator end() {
        return begin() + dim;
    }

    const_iterator begin() const {
        return expression.begin() + start;
    }

    const_iterator end() const {
        return begin() + dim;
    }

    difference_type size() const {
        return dim;
    }

    difference_type stride() const {
        return 1;
    }

    expression_type& expression;
    difference_type start;
    difference_type dim;
};

template<class Expression>
struct VindexRangeNonunitStride :
    public InPlaceOps<VindexRangeNonunitStride<Expression>, typename Expression::value_type> {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    typedef typename expression_type::reference reference;
    typedef typename expression_type::const_reference const_reference;
    typedef typename expression_type::pointer pointer;
    typedef typename expression_type::const_pointer const_pointer;

    typedef InPlaceOps<VindexRangeNonunitStride, value_type> base;

    typedef thrust::transform_iterator<detail::RangeFunctor<difference_type>, thrust::counting_iterator<difference_type> > index_iterator;

    typedef ::thrust::permutation_iterator<typename expression_type::iterator, index_iterator> iterator;
    typedef ::thrust::permutation_iterator<typename expression_type::const_iterator, index_iterator> const_iterator;

    VindexRangeNonunitStride(expression_type& expression,
            difference_type start, difference_type by, difference_type finish)
        : expression(expression), start(start), by(by),
          dim(std::max(difference_type(), difference_type((finish - start) / by + 1)))  // bug here!
    {}

    // Bring in the base assignment operator templates
    using base::operator=;

    // The compiler won't synthesise an assignment operator, since we 
    // have a reference member.  We want something different anyway.
    VindexRangeNonunitStride& operator=(const VindexRangeNonunitStride& other)
    {
        assign_in_place(*this, other);
        return *this;
    }

    reference operator[](difference_type i)
    {
        return expression[by*i + start];
    }

    value_type operator[](difference_type i) const
    {
        return expression[by*i + start];
    }

    iterator begin() {
        return iterator(
            expression.begin(),
            index_iterator(thrust::counting_iterator<difference_type>(0), detail::RangeFunctor<difference_type>(start, by))
        );
    }

    iterator end() {
        return begin() + dim;
    }

    const_iterator begin() const {
        return const_iterator(
            expression.begin(),
            index_iterator(thrust::counting_iterator<difference_type>(0), detail::RangeFunctor<difference_type>(start, by))
        );
    }

    const_iterator end() const {
        return begin() + dim;
    }

    difference_type size() const {
        return dim;
    }

    difference_type stride() const {
        return by;
    }

    expression_type& expression;
    const difference_type start;
    const difference_type by;
    const difference_type dim;
};

template<class Expression, class IndexExpression>
struct VindexI :
    public InPlaceOps<VindexI<Expression, IndexExpression>, typename Expression::value_type> {

    typedef Expression expression_type;
    typedef IndexExpression index_expression;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    typedef typename expression_type::reference reference;
    typedef typename expression_type::const_reference const_reference;
    typedef typename expression_type::pointer pointer;
    typedef typename expression_type::const_pointer const_pointer;

    typedef InPlaceOps<VindexI, value_type> base;

    typedef ::thrust::permutation_iterator<typename expression_type::iterator, typename index_expression::const_iterator> iterator;
    typedef ::thrust::permutation_iterator<typename expression_type::const_iterator, typename index_expression::const_iterator> const_iterator;

    VindexI(expression_type& expression, const index_expression& index) : expression(expression), index(index) {}

    // Bring in the base assignment operator templates
    using base::operator=;

    // The compiler won't synthesise an assignment operator, since we 
    // have a reference member.  We want something different anyway.
    VindexI& operator=(const VindexI& other)
    {
        assign_in_place(*this, other);
        return *this;
    }

    reference operator[](difference_type i)
    {
        return expression[index[i]];
    }

    value_type operator[](difference_type i) const
    {
        return expression[index[i]];
    }

    iterator begin() {
        return iterator(expression.begin(), index.begin());
    }

    iterator end() {
    // which is it - or does it make no difference?
//      return iterator(expression.end(), index.end());
        return iterator(expression.begin(), index.end());
    }

    const_iterator begin() const {
        return const_iterator(expression.begin(), index.begin());
    }

    const_iterator end() const {
    // which is it - or does it make no difference?
//      return const_iterator(expression.end(), index.end());
        return const_iterator(expression.begin(), index.end());
    }

    difference_type size() const {
        return index.size();
    }

    expression_type& expression;
    const index_expression& index;

};


template<class Expression>
struct VdoubleIndexRangeUnitStride :
    public InPlaceOps<VdoubleIndexRangeUnitStride<Expression>, typename Expression::value_type> {

    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    typedef typename expression_type::reference reference;
    typedef typename expression_type::const_reference const_reference;
    typedef typename expression_type::pointer pointer;
    typedef typename expression_type::const_pointer const_pointer;

    typedef thrust::transform_iterator<
        detail::doubleIndexRangeUnitStrideFunctor<difference_type>,
        thrust::counting_iterator<difference_type>
    > index_iterator;

    typedef ::thrust::permutation_iterator<
        typename expression_type::iterator, index_iterator
    > iterator;

    typedef ::thrust::permutation_iterator<
        typename expression_type::const_iterator, index_iterator
    > const_iterator;

    typedef InPlaceOps<VdoubleIndexRangeUnitStride, value_type> base;

    // Bug if empty range
    VdoubleIndexRangeUnitStride(
        expression_type& expression,
        difference_type ld,
        difference_type row_start, difference_type row_finish,
        difference_type col_start, difference_type col_finish)
        : expression(expression),
          ld(ld),
          offset(col_start*ld + row_start),
          rows(row_finish - row_start + 1),
          dim(rows * (col_finish - col_start + 1))
    {}

    // Bring in the base assignment operator templates
    using base::operator=;

    // The compiler won't synthesise an assignment operator, since we 
    // have a reference member.  We want something different anyway.
    VdoubleIndexRangeUnitStride& operator=(const VdoubleIndexRangeUnitStride& other)
    {
        assign_in_place(*this, other);
        return *this;
    }

    reference operator[](difference_type i)
    {
        difference_type row = i % rows;
        difference_type col = i / rows;
        return expression[offset + row + col*ld];
    }

    value_type operator[](difference_type i) const
    {
        difference_type row = i % rows;
        difference_type col = i / rows;
        return expression[offset + row + col*ld];
    }

    iterator begin() {
        return iterator(
            expression.begin(),
            index_iterator(thrust::counting_iterator<difference_type>(0),
                           detail::doubleIndexRangeUnitStrideFunctor<difference_type>(ld, offset, rows))
        );
    }

    iterator end() {
        return begin() + dim;
    }

    const_iterator begin() const {
        return const_iterator(
            expression.begin(),
            index_iterator(thrust::counting_iterator<difference_type>(0),
                           detail::doubleIndexRangeUnitStrideFunctor<difference_type>(ld, offset, rows))
        );
    }

    const_iterator end() const {
        return begin() + dim;
    }

    difference_type size() const {
        return dim;
    }

    expression_type& expression;
    difference_type ld;
    difference_type offset;
    difference_type rows;
    difference_type dim;
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
