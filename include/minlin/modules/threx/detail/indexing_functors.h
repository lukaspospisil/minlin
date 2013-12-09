/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_DETAIL_INDEXING_FUNCTORS_H
#define THREX_DETAIL_INDEXING_FUNCTORS_H

namespace minlin {

namespace threx {

namespace detail {

template<typename T>
struct doubleIndexRangeUnitStrideFunctor {
    typedef T value_type;
    typedef T result_type;
    value_type ld;
    value_type offset;
    value_type rows;
    doubleIndexRangeUnitStrideFunctor(value_type ld, value_type offset, value_type rows)
        : ld(ld), offset(offset), rows(rows) {}
	__host__ __device__
    result_type operator()(value_type value) const
    {
        value_type row = value % rows;
        value_type col = value / rows;
        return offset + row + col*ld;
    }
};

} // end namespace detail

} // end namespace threx

} // end namespace minlin

#endif
