/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_STORAGE_H
#define THREX_STORAGE_H

#include "detail/storage.h"

namespace minlin {

namespace threx {

template<class Storage>
class ByValue : private Storage,
                public detail::InPlaceOps<ByValue<Storage>, typename Storage::value_type> {
public:

    // This type has allocating assignment semantics, which means we need to
    // redefine in_place to be false, and redefine the assignment operator
    enum {is_inplace = false};

    typedef Storage storage;
    typedef typename storage::value_type value_type;
    typedef typename storage::difference_type difference_type;
    typedef typename storage::reference reference;
    typedef typename storage::const_reference const_reference;
    typedef typename storage::pointer pointer;
    typedef typename storage::const_pointer const_pointer;
    typedef typename storage::iterator iterator;
    typedef typename storage::const_iterator const_iterator;

    typedef detail::InPlaceOps<ByValue, value_type> base;

    explicit ByValue(difference_type n = 0)
        : storage(n)
    {
        //std::cout << "ByValue(difference_type) ** ALLOCATION **" << std::endl;
    }

    ByValue(const ByValue& other)
        : storage(other)
    {
        //std::cout << "ByValue(const ByValue&) ** ALLOCATION **" << std::endl;
    }

    template<typename OtherStorage>
    ByValue(const ByValue<OtherStorage>& other)
        : storage(other)
    {
        //std::cout << "ByValue(const ByValue<OtherStorage>&) ** ALLOCATION **" << std::endl;
    }

    template<typename Expression>
    explicit ByValue(const Expression& expression)
        : storage(expression.size())
    {
        //std::cout << "ByValue(const Expression&) ** ALLOCATION **" << std::endl;
        base::operator=(expression);    // in-place assignment
    }

    // Allocating assignment
    ByValue& operator=(const ByValue& other)
    {
        //std::cout << "ByValue::operator=(const ByValue&) ** ALLOCATION **" << std::endl;
        storage::operator=(other);      // allocating assignment
        return *this;
    }

    // Allocating assignment
    // (Handles host <-> device transfers among other things)
    template<typename OtherStorage>
    ByValue& operator=(const ByValue<OtherStorage>& other)
    {
        //std::cout << "ByValue::operator=(const ByValue<OtherStorage>&) ** ALLOCATION **" << std::endl;
        storage::operator=(other);      // allocating assignment
        return *this;
    }

    // Allocating assignment
    template<typename Expression>
    typename detail::expression_enabler<Expression::is_expression, ByValue&>::type
    operator=(const Expression& expression)
    {
        //std::cout << "ByValue::operator=(const Expression&) ** ALLOCATION **" << std::endl;
        storage::resize(expression.size());
        base::operator=(expression);     // in-place assignment
        return *this;
    }

    // Note: no operator=(value_type)

    using storage::operator[];
    using storage::begin;
    using storage::end;
    using storage::size;
//    using storage::resize;
    using storage::data;

    template<typename OtherStorage>
    friend class ByValue;

};

template<class Storage>
class ByReference : public detail::InPlaceOps<ByReference<Storage>, typename Storage::value_type> {
public:

    typedef Storage storage_type;
    typedef typename storage_type::value_type value_type;
    typedef typename storage_type::difference_type difference_type;
    typedef typename storage_type::reference reference;
    typedef typename storage_type::const_reference const_reference;
    typedef typename storage_type::pointer pointer;
    typedef typename storage_type::const_pointer const_pointer;
    typedef typename storage_type::iterator iterator;
    typedef typename storage_type::const_iterator const_iterator;

    // Allow implicit conversion since it's quite convenient to have
    // Vector< ByReference<vector_type> > v(my_vector)
    // rather than
    // Vector< ByReference<vector_type> > v((ByReference<vector_type>(v)))
    ByReference(Storage& storage)
        : storage(storage)
    {
        //std::cout << "ByReference(Storage)" << std::endl;
    }

    reference operator[](difference_type i)
    {
        return storage[i];
    }

    value_type operator[](difference_type i) const
    {
        return storage[i];
    }

    iterator begin() {
        return storage.begin();
    }

    const_iterator begin() const {
        return storage.begin();
    }

    iterator end() {
        return storage.end();
    }
    
    const_iterator end() const {
        return storage.end();
    }

    difference_type size() const {
        return storage.size();
    }

private:
    // No (allocating) assignment
    ByReference& operator=(const ByReference&);

    storage_type& storage;
};

template<typename Iterator>
class Range :
    private detail::RangeWrapper<Iterator>,
    public  detail::InPlaceOps<Range<Iterator>,
                               typename std::iterator_traits<Iterator>::value_type> {
public:

    typedef Iterator iterator;
    typedef iterator const_iterator;

    typedef detail::RangeWrapper<iterator> range_wrapper;

    typedef typename std::iterator_traits<iterator>::value_type value_type;
    typedef typename std::iterator_traits<iterator>::difference_type difference_type;
    typedef typename std::iterator_traits<iterator>::reference reference;
    typedef reference const_reference;
    typedef typename std::iterator_traits<iterator>::pointer pointer;
    typedef pointer const_pointer;

    Range(iterator first, iterator last)
        : range_wrapper(first, last)
     {}

    using range_wrapper::operator[];
    using range_wrapper::begin;
    using range_wrapper::end;
    using range_wrapper::size;

private:
    // No (allocating) assignment
    Range& operator=(const Range&);

};

} // end namespace threx

} // end namespace minlin

#endif
