/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
// Todo:

// fix difference_type/all overload clash by adding int overload

// investigate overloads of assign_in_place for direct copying host <-> device

// review all code

// swap

// check all empty subranges

// all matrix subranges

// one-argument size, and two-argument size for vector

// vector end (and arithmetic?)

// fix bug in range, such as 1:2:0, since we get size = (0-1) / 2 + 1 = 0 + 1 = 1

// dimension-aware IO

// transpose

// find
// first call count to get size, then copy_if based on counting iterator/range and bool vector

// sort, unique

// min, max, minmax

// set operations

// operator() on bool vector
// need to specialise on bool so that it's not treated as 0 and 1 integer indexing

// Done:
// explicit constructors; e.g. ByValue and others

// matrix and vector assignment crossovers

// v(all) = k
// fill

// unary function operators

// mul
// multiple argument mul

// arithmetic operators
// atan2, pow
// unary plus and minus

// compound assignment operators

// relational operators
// logical operators

// vector orientation

// vector range indexing

// change LValue / ByValue / etc. to remove hacky assignment_conforms_with

// range checking

// check where rows() and cols() should go
// replace row_ with orientation_, and include orientation(), and replace make_vector with that, and lose rows() and cols() from all expressions

// redesign lvalue and inplace as separate types, and sort out all the in-place stuff
// idea: inplace base using crtp, defines op=, op+=, etc. which casts to derived and calls assign_in_place, etc.

// move reduction operators from unary_operators to reduction_operators


#ifndef MINLIN_H
#define MINLIN_H

#include "vector.h"
#include "vector_operators.h"
#include "vector_functions.h"
#include "vector_properties.h"
#include "vector_io.h"

#include "matrix.h"
#include "matrix_operators.h"
#include "matrix_functions.h"
#include "matrix_io.h"

#endif
