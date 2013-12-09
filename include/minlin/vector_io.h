/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef MINLIN_VECTOR_IO_H
#define MINLIN_VECTOR_IO_H

#include "vector.h"

#include <ostream>

namespace minlin {

// Output streaming
// ****************

template<class Expression>
std::ostream& operator<<(std::ostream& os, const Vector<Expression>& v)
{
    os << ' ';
    for (typename Vector<Expression>::difference_type i = 0; i < v.size(); ++i)
    {
        os << v(i) << ' ';
    }
    return os;
}

} // end namespace minlin

#endif
