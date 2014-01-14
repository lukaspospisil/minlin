#pragma once

// blas routines for cuda and mkl
#include <cublas_v2.h>
#include <mkl.h>

#include <string>

// implementation of gemv kernel
// uses blas routines provided by either MKL or CUBLAS
//
// Ben Cumming


namespace minlin {
namespace threx {
namespace detail {

//////////////////////////////////////////////////////////////
// wrapper for gemv call to MKL
//////////////////////////////////////////////////////////////
// TODO : add checks that ensure only double and float are accepted

// overloaded for double
bool gemv_host (
    double const* A, double const* x, double* y,
    double alpha, double beta,
    int m, int n,
    int incx, int incy, int lda,
    char trans)
{
    dgemv(&trans, &m, &n, &alpha, const_cast<double*>(A), &lda, const_cast<double*>(x), &incx, &beta, y, &incy);
    return true;
}

// overloaded for float
bool gemv_host (
    float const* A, float const* x, float* y,
    float alpha, float beta,
    int m, int n,
    int incx, int incy, int lda,
    char trans)
{
    sgemv(&trans, &m, &n, &alpha, const_cast<float*>(A), &lda, const_cast<float*>(x), &incx, &beta, y, &incy);
    return true;
}

template <typename T>
struct print_traits {
    static std::string print(){ return std::string("unknown"); };
};
template <>
struct print_traits<float> {
    static std::string print(){ return std::string("float"); };
};
template <>
struct print_traits<double> {
    static std::string print(){ return std::string("double"); };
};

} //namespace detail
} //namespace threx
} //namespace minlin
