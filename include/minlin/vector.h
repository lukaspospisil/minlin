/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_VECTOR_H
#define MINLIN_VECTOR_H

#include "detail/forward.h"
#include "detail/helpers.h"

#include <cassert>

namespace minlin {


// Vector is one of the main types in the library.

// It is a model of a vector in the linear algebra sense.

// The implementation is a standard expression template approach, where
// the Vector class provides the outer type and associated operators,
// while forwarding the calls to the underlying Expression type which
// provides the implementations, and on which Vector itself is templated.
// See the main documentation for what operations the Expression type should
// support.

// Besides the expression member, Vector also has an orientation member,
// which is used purely to ensure conformance in expressions with other
// vectors and matrices (see MINLIN_DEBUG macro).

// Range and conformance checking is optionally enabled through the
// MINLIN_DEBUG macro.



template<class Expression>
class Vector {
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

    // Create empty vector (size() == 0)
    Vector()
        : expression_(), orientation_(ColumnOriented) {}

    // Create vector of specified size with optionally specified orientation
    // (default is column oriented)
    explicit Vector(difference_type n, orientation_type orientation = ColumnOriented)
        : expression_(n), orientation_(orientation)
    {
        #ifdef MINLIN_DEBUG
        std::cout << "Vector(difference_type)" << std::endl;
        #endif
    }

    // Create vector as a copy of another of the same type
    Vector(const Vector& other)
        : expression_(other.expression()), orientation_(other.orientation())
    {
        #ifdef MINLIN_DEBUG
        std::cout << "Vector(const Vector&)" << std::endl;
        #endif
    }

    // Create vector as a copy of another of a different type
    // "Another" type is any type not comprising the same Expression type.
    template<class OtherExpression>
    Vector(const Vector<OtherExpression>& other)
        : expression_(other.expression()), orientation_(other.orientation())
    {
        #ifdef MINLIN_DEBUG
        std::cout << "Vector(const Vector<OtherExpression>&)" << std::endl;
        #endif
    }

    // Create a vector as a copy of a matrix.
    // The matrix must be a single row or column.
    template<class OtherExpression>
    explicit Vector(const Matrix<OtherExpression>& other)
        : expression_(other.expression()),
          orientation_(other.cols() == 1 ? ColumnOriented : RowOriented)
    {
        #ifdef MINLIN_DEBUG
        std::cout << "Vector(const Matrix<OtherExpression>&)" << std::endl;
        assert(other.rows() == 1 || other.cols() == 1);
        #endif
    }

    // Create a vector as a window into existing storage.
    // No copy is made, so any modifications affect the existing storage.
    Vector(const expression_type& expression, orientation_type orientation = ColumnOriented)
        :expression_(expression), orientation_(orientation)
    {
        #ifdef MINLIN_DEBUG
        std::cout << "Vector(const Expression&)" << std::endl;
        assert(orientation == RowOriented || orientation == ColumnOriented);
        #endif
    }

    // Assignment
    // **********

    // Assign to a vector from a vector of the same type.
    // Expressions of the form lhs(all) = rhs assign in-place, and must conform in size
    // (but not necessarily orientation).
    // Expressions of the form lhs = rhs allocate new storage.
    Vector& operator=(const Vector& other)
    {

        #ifdef MINLIN_DEBUG
        std::cout << "Vector::operator=(const Vector&)" << std::endl;
        // assert conformance in size (don't) if the assignment is in-place (allocating)
        detail::assert_conformance<expression_type::is_inplace>::assignment(*this, other);
        #endif

        // assign dimensions (don't) if the assignment is allocating (in-place)
        detail::assign_dimensions_to_v<expression_type::is_inplace>(*this, other);

        expression() = other.expression();
        return *this;
    }

    // Assign to a vector from a vector of different type.
    // Expressions of the form lhs(all) = rhs assign in-place, and must conform in size
    // (but not necessarily orientation)
    // Expressions of the form lhs = rhs allocate new storage.
    template<class OtherExpression>
    Vector& operator=(const Vector<OtherExpression>& other)
    {

        #ifdef MINLIN_DEBUG
        std::cout << "Vector::operator=(const Vector<OtherExpression>&)" << std::endl;
        // assert conformance in size (don't) if the assignment is in-place (allocating)
        detail::assert_conformance<expression_type::is_inplace>::assignment(*this, other);
        #endif

        // assign dimensions (don't) if the assignment is allocating (in-place)
        detail::assign_dimensions_to_v<expression_type::is_inplace>(*this, other);

        expression() = other.expression();
        return *this;
    }

    // Assign to a vector from a matrix.
    // Expressions of the form lhs(all) = rhs assign in-place, and must conform in size
    // but not necessarily dimension (i.e. you can assign a 2x2 matrix to a 4x1 vector)
    // Expressions of the form lhs = rhs allocate new storage, and the matrix must be
    // a single row or column.
    template<class OtherExpression>
    Vector& operator=(const Matrix<OtherExpression>& other)
    {

        #ifdef MINLIN_DEBUG
        std::cout << "Vector::operator=(const Matrix<OtherExpression>&)" << std::endl;
        // assert conformance in size if the assigmment is in-place
        // assert matrix is a single row or column if the assignment is allocating
        detail::assert_conformance<expression_type::is_inplace>::assignment_m_to_v(*this, other);
        #endif

        // assign dimensions (don't) if the assignment is allocating (in-place)
        detail::assign_dimensions_to_v<expression_type::is_inplace>(*this, other);

        expression() = other.expression();
        return *this;
    }

    // Assign to each element of the vector the same value.
    // Only valid for an expression of the form lhs(all) = value.
    Vector& operator=(value_type value)
    {
        expression() = value;
        return *this;
    }

    // Compound assignment
    // *******************

    // Todo: documentation for compound assignment
    
    template<class OtherExpression>
    Vector& operator+=(const Vector<OtherExpression>& other)
    {
        #ifdef MINLIN_DEBUG
        std::cout << "Vector::operator+=(const Vector<OtherExpression>&)" << std::endl;
        detail::assert_conformance<expression_type::is_inplace>::compound(*this, other);
        #endif
        expression() += other.expression();
        return *this;
    }

    template<class OtherExpression>
    Vector& operator-=(const Vector<OtherExpression>& other)
    {
        #ifdef MINLIN_DEBUG
        std::cout << "Vector::operator+=(const Vector<OtherExpression>&)" << std::endl;
        detail::assert_conformance<expression_type::is_inplace>::compound(*this, other);
        #endif
        expression() -= other.expression();
        return *this;
    }

    template<class OtherExpression>
    Vector& operator*=(const Vector<OtherExpression>& other)
    {
        #ifdef MINLIN_DEBUG
        std::cout << "Vector::operator+=(const Vector<OtherExpression>&)" << std::endl;
        detail::assert_conformance<expression_type::is_inplace>::compound(*this, other);
        #endif
        expression() *= other.expression();
        return *this;
    }

    template<class OtherExpression>
    Vector& operator/=(const Vector<OtherExpression>& other)
    {
        #ifdef MINLIN_DEBUG
        std::cout << "Vector::operator+=(const Vector<OtherExpression>&)" << std::endl;
        detail::assert_conformance<expression_type::is_inplace>::compound(*this, other);
        #endif
        expression() /= other.expression();
        return *this;
    }

    // Todo: Compound assignment to vector from matrix

    Vector& operator+=(value_type value)
    {
        expression() += value;
        return *this;
    }

    Vector& operator-=(value_type value)
    {
        expression() -= value;
        return *this;
    }

    Vector& operator*=(value_type value)
    {
        expression() *= value;
        return *this;
    }

    Vector& operator/=(value_type value)
    {
        expression() /= value;
        return *this;
    }

    // Indexing
    // ********

    // A variety of indexing options is provided.  All are zero-based.

    // Single index (non-const, by reference)
    reference operator()(difference_type i)
    {
        #ifdef MINLIN_DEBUG
        assert(i >= 0 && i < size());
        #endif
        return expression()[i];
    }

    // Single index (const, by value)
    value_type operator()(difference_type i) const
    {
        #ifdef MINLIN_DEBUG
        assert(i >= 0 && i < size());
        #endif
        return expression()[i];
    }

    // "All" index (non-const, by vector of helper type)
    Vector<typename Expression::template inplace<Expression>::type>
    operator()(detail::all_type)
    {
        return make_vector(inplace(expression()), ColumnOriented);
    }

    // "All" index (const, by const vector of helper type)
    // Todo: investigate if it's better to just return const vector
    const Vector<typename Expression::template inplace<Expression>::type>
    operator()(detail::all_type) const
    {
        return make_vector(inplace(expression()), ColumnOriented);
    }

    // Note: presently detail::all_type is defined as a pointer to function type.
    // This is to remove ambiguity with a global all function that CUDA brings in.
    // However that presents its own ambiguity with the call v(0), when difference_type
    // is not int (0 can convert to either integer 0 or null pointer constant).
    // These next two overloads resolve that ambiguity

    // Todo: disable these overload when difference_type and int are the same

    // Single index (non-const, by reference)
    reference operator()(int i)
    {
        return operator()(difference_type(i));
    }

    // Single index (const, by value)
    value_type operator()(int i) const
    {
        return operator()(difference_type(i));
    }

    // Unit stride range index (non-const, by vector of helper type)
    Vector<typename Expression::template index_range_unit_stride<Expression>::type>
    operator()(difference_type start, difference_type finish)
    {
        #ifdef MINLIN_DEBUG
        assert(start >= 0 && start < size());
        assert(finish >= 0 && finish < size());
        #endif
        return make_vector(index_range_unit_stride(expression(), start, finish), orientation());
    }

    // Unit stride range index (const, by const vector of helper type)
    const Vector<typename Expression::template index_range_unit_stride<const Expression>::type>
    operator()(difference_type start, difference_type finish) const
    {
        #ifdef MINLIN_DEBUG
        assert(start >= 0 && start < size());
        assert(finish >= 0 && finish < size());
        #endif
        return make_vector(index_range_unit_stride(expression(), start, finish), orientation());
    }

    // Non-unit stride range index (non-const, by vector of helper type)
    Vector<typename Expression::template index_range_nonunit_stride<Expression>::type>
    operator()(difference_type start, difference_type by, difference_type finish)
    {
        #ifdef MINLIN_DEBUG
        assert(start >= 0 && start < size());
        // Need to be careful with the upper check, since something like v(0,2,5) is fine even if v.size() == 5
//      assert(finish >= 0 && finish < size());
        assert(finish >= 0 && start + ((finish - start) /  by) * by < size());
        #endif
        return make_vector(index_range_nonunit_stride(expression(), start, by, finish), orientation());
    }

    // Non-unit stride range index (const, by const vector of helper type)
    const Vector<typename Expression::template index_range_nonunit_stride<const Expression>::type>
    operator()(difference_type start, difference_type by, difference_type finish) const
    {
        #ifdef MINLIN_DEBUG
        assert(start >= 0 && start < size());
        // Need to be careful with the upper check, since something like v(0,2,5) is fine if v.size() == 5
//      assert(finish >= 0 && finish < size());
        assert(finish >= 0 && start + ((finish - start) /  by) * by < size());
        #endif
        return make_vector(index_range_nonunit_stride(expression(), start, by, finish), orientation());
    }

    // Indirect index (non-const, by vector of helper type)
    template<class IndexExpression>
    Vector<typename Expression::template indirect_index<Expression, IndexExpression>::type>
    operator()(const Vector<IndexExpression>& idx)
    {
        #ifdef MINLIN_DEBUG
        assert(all_of(idx >= 0 && idx < size()));
        #endif
        return make_vector(indirect_index(expression(), idx.expression()), orientation());
    }

    // Indirect index (const, by const vector of helper type)
    template<class IndexExpression>
    const Vector<typename Expression::template indirect_index<const Expression, IndexExpression>::type>
    operator()(const Vector<IndexExpression>& idx) const
    {
        #ifdef MINLIN_DEBUG
        assert(all_of(idx >= 0 && idx < size()));
        #endif
        return make_vector(indirect_index(expression(), idx.expression()), orientation());
    }

    // Properties
    // **********

    // All self-explanatory

    orientation_type orientation() const {
        return orientation_;
    }

    difference_type size() const {
        return expression().size();
    }

    difference_type rows() const {
        return orientation() == RowOriented ? 1 : size();
    }

    difference_type cols() const {
        return orientation() == ColumnOriented ? 1 : size();
    }

    expression_type& expression() {
        return expression_;
    }

    const expression_type& expression() const {
        return expression_;
    }

private:
    expression_type expression_;
    orientation_type orientation_;

    template<bool InPlace>
    friend class detail::assign_dimensions_to_v;

};

// Usual make_something helper function to do template argument deduction
template<typename Expression>
Vector<Expression> make_vector(const Expression& expression,
                               typename Vector<Expression>::orientation_type orientation)
{
    return Vector<Expression>(expression, orientation);
}

} // end namespace minlin

#endif
