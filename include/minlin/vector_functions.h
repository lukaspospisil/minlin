/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_VECTOR_FUNCTIONS_H
#define MINLIN_VECTOR_FUNCTIONS_H

#include "vector.h"

namespace minlin {

template<class Expression>
bool any_of(const Vector<Expression>& vec)
{
    return any_of(vec.expression());
}

template<class Expression>
bool all_of(const Vector<Expression>& vec)
{
    return all_of(vec.expression());
}

} // end namespace minlin

#endif
