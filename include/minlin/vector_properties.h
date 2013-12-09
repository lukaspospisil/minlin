/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_VECTOR_PROPERTIES_H
#define MINLIN_VECTOR_PROPERTIES_H

#include "vector.h"

namespace minlin {

template<class Expression>
typename Expression::difference_type
length(const Vector<Expression>& vec)
{
    return vec.size();
}

template<class Expression>
typename Expression::difference_type
numel(const Vector<Expression>& vec)
{
    return vec.size();
}

} // end namespace minlin

#endif
