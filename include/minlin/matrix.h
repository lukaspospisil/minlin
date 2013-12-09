/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_MATRIX_H
#define MINLIN_MATRIX_H

#include "detail/forward.h"
#include "detail/helpers.h"

#include <cassert>

namespace minlin {

// Matrix is one of the main types in the library.

// It is a model of a matrix in the linear algebra sense.

// The implementation is a standard expression template approach, where
// the Matrix class provides the outer type and associated operators,
// while forwarding the calls to the underlying Expression type which
// provides the implementations, and on which Matrix itself is templated.
// See the main documentation for what operations the Expression type should
// support.

// Besides the expression member, Matrix also has rows and columns members,
// which are used purely to ensure conformance in expressions with other
// vectors and matrices (see MINLIN_DEBUG macro).

// Range and conformance checking is optionally enabled through the
// MINLIN_DEBUG macro.

// Todo: write rest of documentation

template<class Expression>
class Matrix {
public:
    typedef Expression expression_type;

    typedef typename expression_type::value_type value_type;
    typedef typename expression_type::difference_type difference_type;
    typedef typename expression_type::pointer pointer;
    typedef typename expression_type::const_pointer const_pointer;
    typedef typename expression_type::reference reference;
    typedef typename expression_type::const_reference const_reference;
    typedef typename expression_type::iterator iterator;
    typedef typename expression_type::const_iterator const_iterator;

    typedef VectorOrientation orientation_type;

    // Construction
    // ************

    Matrix()
        : expression_(), rows_(0), cols_(0) {}

    Matrix(difference_type m, difference_type n)
        : expression_(m*n), rows_(m), cols_(n)
    {
        //std::cout << "Matrix(difference_type, difference_type)" << std::endl;
    }

    Matrix(const Matrix& other)
        : expression_(other.expression()), rows_(other.rows()), cols_(other.cols())
    {
        //std::cout << "Matrix(const Matrix&)" << std::endl;
    }

    template<class OtherExpression>
    Matrix(const Matrix<OtherExpression>& other)
        : expression_(other.expression()), rows_(other.rows()), cols_(other.cols())
    {
        //std::cout << "Matrix(const Matrix<OtherExpression>&)" << std::endl;
    }

    template<class OtherExpression>
    explicit Matrix(const Vector<OtherExpression>& other)
        : expression_(other.expression()), rows_(other.rows()), cols_(other.cols())
    {
        //std::cout << "Matrix(const Vector<OtherExpression>&)" << std::endl;
    }

    // This can be used to create a matrix from an existing expression
    // e.g. pre-allocated storage by value or reference
    Matrix(const expression_type& expression, difference_type m, difference_type n)
        :expression_(expression), rows_(m), cols_(n)
    {
        #ifdef MINLIN_DEBUG
        assert(rows() * cols() == size());
        #endif
        //std::cout << "Matrix(const Expression&)" << std::endl;
    }

    // Assignment
    // **********

    Matrix& operator=(const Matrix& other)
    {
        //std::cout << "Matrix::operator=(const Matrix&)" << std::endl;
        #ifdef MINLIN_DEBUG
        detail::assert_conformance<expression_type::is_inplace>::assignment(*this, other);
        #endif
        detail::assign_dimensions_to_m<expression_type::is_inplace>(*this, other);
        expression() = other.expression();
        return *this;
    }

    template<class OtherExpression>
    Matrix& operator=(const Matrix<OtherExpression>& other)
    {
        //std::cout << "Matrix::operator=(const Matrix<OtherExpression>&)" << std::endl;
        #ifdef MINLIN_DEBUG
        detail::assert_conformance<expression_type::is_inplace>::assignment(*this, other);
        #endif
        detail::assign_dimensions_to_m<expression_type::is_inplace>(*this, other);
        expression() = other.expression();
        return *this;
    }

    template<class OtherExpression>
    Matrix& operator=(const Vector<OtherExpression>& other)
    {
        //std::cout << "Matrix::operator=(const Vector<OtherExpression>&)" << std::endl;
        #ifdef MINLIN_DEBUG
        detail::assert_conformance<expression_type::is_inplace>::assignment(*this, other);
        #endif
        detail::assign_dimensions_to_m<expression_type::is_inplace>(*this, other);
        expression() = other.expression();
        return *this;
    }

    Matrix& operator=(value_type value)
    {
        expression() = value;
        return *this;
    }

    // Compound assignment
    // *******************
    template<class OtherExpression>
    Matrix& operator+=(const Matrix<OtherExpression>& other)
    {
        //std::cout << "Matrix::operator+=(const Matrix<OtherExpression>&)" << std::endl;
        #ifdef MINLIN_DEBUG
        detail::assert_conformance<expression_type::is_inplace>::compound(*this, other);
        #endif
        expression() += other.expression();
        return *this;
    }

    template<class OtherExpression>
    Matrix& operator-=(const Matrix<OtherExpression>& other)
    {
        //std::cout << "Matrix::operator+=(const Matrix<OtherExpression>&)" << std::endl;
        #ifdef MINLIN_DEBUG
        detail::assert_conformance<expression_type::is_inplace>::compound(*this, other);
        #endif
        expression() -= other.expression();
        return *this;
    }

    template<class OtherExpression>
    Matrix& operator*=(const Matrix<OtherExpression>& other)
    {
        //std::cout << "Matrix::operator+=(const Matrix<OtherExpression>&)" << std::endl;
        #ifdef MINLIN_DEBUG
        detail::assert_conformance<expression_type::is_inplace>::compound(*this, other);
        #endif
        expression() *= other.expression();
        return *this;
    }

    template<class OtherExpression>
    Matrix& operator/=(const Matrix<OtherExpression>& other)
    {
        //std::cout << "Matrix::operator+=(const Matrix<OtherExpression>&)" << std::endl;
        #ifdef MINLIN_DEBUG
        detail::assert_conformance<expression_type::is_inplace>::compound(*this, other);
        #endif
        expression() /= other.expression();
        return *this;
    }

    Matrix& operator+=(value_type value)
    {
        expression() += value;
        return *this;
    }

    Matrix& operator-=(value_type value)
    {
        expression() -= value;
        return *this;
    }

    Matrix& operator*=(value_type value)
    {
        expression() *= value;
        return *this;
    }

    Matrix& operator/=(value_type value)
    {
        expression() /= value;
        return *this;
    }

    // Indexing
    // ********

    reference operator()(difference_type i, difference_type j)
    {
        #ifdef MINLIN_DEBUG
        assert(i >= 0 && i < rows());
        assert(j >= 0 && j < cols());
        #endif
        return expression()[i + j*rows()];
    }

    value_type operator()(difference_type i, difference_type j) const
    {
        #ifdef MINLIN_DEBUG
        assert(i >= 0 && i < rows());
        assert(j >= 0 && j < cols());
        #endif
        return expression()[i + j*rows()];
    }

    // This is to resolve ambiguity between all_type and difference_type
    // Todo: fix this for the case where difference_type and int are the same
    reference operator()(int i, difference_type j)
    {
        return operator()(difference_type(i), j);
    }
    reference operator()(difference_type i, int j)
    {
        return operator()(i, difference_type(j));
    }
    reference operator()(int i, int j)
    {
        return operator()(difference_type(i), difference_type(j));
    }

    // This is to resolve ambiguity between all_type and difference_type
    // Todo: fix this for the case where difference_type and int are the same
    value_type operator()(int i, difference_type j) const
    {
        return operator()(difference_type(i), j);
    }
    value_type operator()(difference_type i, int j) const
    {
        return operator()(i, difference_type(j));
    }
    value_type operator()(int i, int j) const
    {
        return operator()(difference_type(i), difference_type(j));
    }
    
    reference operator()(difference_type p)
    {
        #ifdef MINLIN_DEBUG
        assert(p >= 0 && p < rows()*cols());
        #endif
        return expression()[p];
    }

    value_type operator()(difference_type p) const
    {
        #ifdef MINLIN_DEBUG
        assert(p >= 0 && p < rows()*cols());
        #endif
        return expression()[p];
    }

    // This is to resolve ambiguity between all_type and difference_type
    // Todo: fix this for the case where difference_type and int are the same
    reference operator()(int p)
    {
        return operator()(difference_type(p));
    }

    // This is to resolve ambiguity between all_type and difference_type
    // Todo: fix this for the case where difference_type and int are the same
    value_type operator()(int p) const
    {
        return operator()(difference_type(p));
    }

    Vector<typename Expression::template inplace<Expression>::type>
    operator()(detail::all_type)
    {
        return make_vector(inplace(expression()), ColumnOriented);
    }

    // Could we just return const Matrix instead?
    const Vector<typename Expression::template inplace<const Expression>::type>
    operator()(detail::all_type) const
    {
        return make_vector(inplace(expression()), ColumnOriented);
    }

    Vector<typename Expression::template index_range_nonunit_stride<Expression>::type>
    operator()(difference_type i, detail::all_type)
    {
        #ifdef MINLIN_DEBUG
        assert(i >= 0 && i < rows());
        #endif
        return make_vector(index_range_nonunit_stride(expression(), i, rows(), size() - 1), RowOriented);
    }

    const Vector<typename Expression::template index_range_nonunit_stride<const Expression>::type>
    operator()(difference_type i, detail::all_type) const
    {
        #ifdef MINLIN_DEBUG
        assert(i >= 0 && i < rows());
        #endif
        return make_vector(index_range_nonunit_stride(expression(), i, rows(), size() - 1), RowOriented);
    }

    Vector<typename Expression::template index_range_unit_stride<Expression>::type>
    operator()(detail::all_type, difference_type j)
    {
        #ifdef MINLIN_DEBUG
        assert(j >= 0 && j < cols());
        #endif
        return make_vector(index_range_unit_stride(expression(), j*rows(), (j+1)*rows()-1), ColumnOriented);
    }

    const Vector<typename Expression::template index_range_unit_stride<const Expression>::type>
    operator()(detail::all_type, difference_type j) const
    {
        #ifdef MINLIN_DEBUG
        assert(j >= 0 && j < cols());
        #endif
        return make_vector(index_range_unit_stride(expression(), j*rows(), (j+1)*rows()-1), ColumnOriented);
    }

    Matrix<typename Expression::template double_index_range_unit_stride<Expression>::type>
    operator()(difference_type row_start, difference_type row_finish,
               difference_type col_start, difference_type col_finish)
    {
        #ifdef MINLIN_DEBUG
        assert(row_start >= 0 && row_start < rows());
        assert(row_finish >= 0 && row_finish < rows());
        assert(col_start >= 0 && col_start < cols());
        assert(col_finish >= 0 && col_finish < cols());
        #endif
        difference_type new_rows = row_finish - row_start + 1;
        difference_type new_cols = col_finish - col_start + 1;
        return make_matrix(double_index_range_unit_stride(
            expression(), rows(),
            row_start, row_finish, col_start, col_finish), new_rows, new_cols
        );
    }

    const Matrix<typename Expression::template double_index_range_unit_stride<const Expression>::type>
    operator()(difference_type row_start, difference_type row_finish,
               difference_type col_start, difference_type col_finish) const
    {
        #ifdef MINLIN_DEBUG
        assert(row_start >= 0 && row_start < rows());
        assert(row_finish >= 0 && row_finish < rows());
        assert(col_start >= 0 && col_start < cols());
        assert(col_finish >= 0 && col_finish < cols());
        #endif
        difference_type new_rows = row_finish - row_start + 1;
        difference_type new_cols = col_finish - col_start + 1;
        return make_matrix(double_index_range_unit_stride(
            expression(), rows(),
            row_start, row_finish, col_start, col_finish), new_rows, new_cols
        );
    }

    Matrix<typename Expression::template double_index_range_unit_stride<Expression>::type>
    operator()(detail::all_type,
               difference_type col_start, difference_type col_finish)
    {
        #ifdef MINLIN_DEBUG
        assert(col_start >= 0 && col_start < cols());
        assert(col_finish >= 0 && col_finish < cols());
        #endif
        difference_type new_cols = col_finish - col_start + 1;
        return make_matrix(double_index_range_unit_stride(
            expression(), rows(),
            0, rows()-1, col_start, col_finish), rows(), new_cols
        );
    }

    const Matrix<typename Expression::template double_index_range_unit_stride<const Expression>::type>
    operator()(detail::all_type,
               difference_type col_start, difference_type col_finish) const
    {
        #ifdef MINLIN_DEBUG
        assert(col_start >= 0 && col_start < cols());
        assert(col_finish >= 0 && col_finish < cols());
        #endif
        difference_type new_cols = col_finish - col_start + 1;
        return make_matrix(double_index_range_unit_stride(
            expression(), rows(),
            0, rows()-1, col_start, col_finish), rows(), new_cols
        );
    }

    Matrix<typename Expression::template double_index_range_unit_stride<Expression>::type>
    operator()(difference_type row_start, difference_type row_finish,
               detail::all_type)
    {
        #ifdef MINLIN_DEBUG
        assert(row_start >= 0 && row_start < rows());
        assert(row_finish >= 0 && row_finish < rows());
        #endif
        difference_type new_rows = row_finish - row_start + 1;
        return make_matrix(double_index_range_unit_stride(
            expression(), rows(),
            row_start, row_finish, 0, cols()-1), new_rows, cols()
        );
    }

    const Matrix<typename Expression::template double_index_range_unit_stride<const Expression>::type>
    operator()(difference_type row_start, difference_type row_finish,
               detail::all_type) const
    {
        #ifdef MINLIN_DEBUG
        assert(row_start >= 0 && row_start < rows());
        assert(row_finish >= 0 && row_finish < rows());
        #endif
        difference_type new_rows = row_finish - row_start + 1;
        return make_matrix(double_index_range_unit_stride(
            expression(), rows(),
            row_start, row_finish, 0, cols()-1), new_rows, cols()
        );
    }

    // Properties
    // **********

    difference_type rows() const {
        return rows_;
    }
    
    difference_type cols() const {
        return cols_;
    }

    difference_type size() const {
        return expression().size();
    }

    orientation_type orientation() const {
        #ifdef MINLIN_DEBUG
        assert(rows() == 1 || cols() == 1);
        #endif
        return rows() == 1 ? RowOriented : ColumnOriented;
    }

    expression_type& expression() {
        return expression_;
    }

    const expression_type& expression() const {
        return expression_;
    }

private:
    expression_type expression_;
    difference_type rows_;
    difference_type cols_;

    template<bool InPlace>
    friend class detail::assign_dimensions_to_m;
};

template<typename Expression>
Matrix<Expression> make_matrix(const Expression& expression,
                               typename Expression::difference_type m,
                               typename Expression::difference_type n)
{
    return Matrix<Expression>(expression, m, n);
}

} // end namespace minlin

#endif
