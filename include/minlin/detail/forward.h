/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_DETAIL_FORWARD_H
#define MINLIN_DETAIL_FORWARD_H

// Forward declaration of vector and matrix types and "make" functions

namespace minlin {

template<typename Expression>
class Vector;

template<typename Expression>
Vector<Expression> make_vector(const Expression& expression,
                               typename Vector<Expression>::orientation_type orientation);

template<typename Expression>
class Matrix;

template<typename Expression>
Matrix<Expression> make_matrix(const Expression& expression,
                               typename Expression::difference_type,
                               typename Expression::difference_type);

} // end namespace minlin

#endif
