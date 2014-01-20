#pragma once

// blas routines for cuda and mkl
#include <cublas_v2.h>
#include <mkl.h>

#include <string>

#include "../blas.h"

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

//////////////////////////////////////////////////////////////
// wrapper for gemv call to cublas
//////////////////////////////////////////////////////////////
// overloaded for double
bool gemv_device (
    double const* A, double const* x, double* y,
    double alpha, double beta,
    int m, int n,
    int incx, int incy, int lda,
    char trans)
{
    cublasHandle_t& handle = CublasState::instance()->handle();
    cublasOperation_t opt = trans=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasStatus_t status =
        cublasDgemv(handle, opt, m, n, &alpha, const_cast<double*>(A), lda, const_cast<double*>(x), incx, &beta, y, incy);
    return (status == CUBLAS_STATUS_SUCCESS);
}

// overloaded for double
bool gemv_device (
    float const* A, float const* x, float* y,
    float alpha, float beta,
    int m, int n,
    int incx, int incy, int lda,
    char trans)
{
    cublasHandle_t& handle = CublasState::instance()->handle();
    cublasOperation_t opt = trans=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasStatus_t status =
        cublasSgemv(handle, opt, m, n, &alpha, const_cast<float*>(A), lda, const_cast<float*>(x), incx, &beta, y, incy);
    return (status == CUBLAS_STATUS_SUCCESS);
}

} //namespace detail
} //namespace threx
} //namespace minlin
