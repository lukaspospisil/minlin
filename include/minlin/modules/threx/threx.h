/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#ifndef THREX_H
#define THREX_H

#include "operators.h"
//#include "storage.h"
#include "functions.h"
#include "host_vector.h"
#include "device_vector.h"
#include "host_matrix.h"
#include "device_matrix.h"
#include "blas.h"

// Some cuda header seems to want to define a bool all(bool) in global scope.
// Unbelievable!  So here's the workaround for now - just use that global all.
// the __device__ keyword added by Ben Cumming to avoid compiler warning
// ... maybe should be a __device__ __host__
extern __device__ bool all(bool);

namespace minlin {

namespace threx {

//using ::minlin::end;
//using ::minlin::all;
using minlin::inf;

} // end namespace threx

} // end namespace minlin

#endif
