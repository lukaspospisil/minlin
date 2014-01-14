/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_INDEXING_OPERATORS_H
#define THREX_INDEXING_OPERATORS_H

#include "detail/expression_types.h"
#include "detail/indexing_expressions.h"

namespace minlin {

namespace threx {

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::InPlace<Expression> >::type
inplace(Expression& expression)
{
    return detail::InPlace<Expression>(expression);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VindexRangeUnitStride<Expression> >::type
index_range_unit_stride(Expression& expression,
        typename Expression::difference_type start,
        typename Expression::difference_type finish)
{
    return detail::VindexRangeUnitStride<Expression>(expression, start, finish);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VindexRangeNonunitStride<Expression> >::type
index_range_nonunit_stride(Expression& expression,
        typename Expression::difference_type start,
        typename Expression::difference_type by,
        typename Expression::difference_type finish)
{
    return detail::VindexRangeNonunitStride<Expression>(expression, start, by, finish);
}

template<class Expression, class IndexExpression>
typename detail::expressions_enabler<Expression::is_expression, IndexExpression::is_expression,
detail::VindexI<Expression, const IndexExpression> >::type
indirect_index(Expression& expression, const IndexExpression& index)
{
    return detail::VindexI<Expression, const IndexExpression>(expression, index);
}

template<class Expression>
typename detail::expression_enabler<Expression::is_expression,
detail::VdoubleIndexRangeUnitStride<Expression> >::type
double_index_range_unit_stride(Expression& expression,
                               typename Expression::difference_type rows,
                               typename Expression::difference_type row_start,
                               typename Expression::difference_type row_end,
                               typename Expression::difference_type col_start,
                               typename Expression::difference_type col_end)
{
    return detail::VdoubleIndexRangeUnitStride<Expression>(
        expression, rows, row_start, row_end, col_start, col_end
    );
}

} // end namespace threx

} // end namespace minlin

#endif
