#pragma once

// blas routines for cuda and mkl
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
#include <cublas_v2.h>
#endif

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
// wrapper for gemm call to MKL
//////////////////////////////////////////////////////////////
// overloaded for double
bool gemm_host (
    double const* A, double const* B, double* C,
    double alpha, double beta,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    char transa, char transb)
{
    // FORTAN interface
    //sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    dgemm(&transa, &transb, &m, &n, &k, &alpha, const_cast<double*>(A), &lda, const_cast<double*>(B), &ldb, &beta, C, &ldc);
    return true;
}

// overloaded for float
bool gemm_host (
    float const* A, float const* B, float* C,
    float alpha, float beta,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    char transa, char transb)
{
    sgemm(&transa, &transb, &m, &n, &k, &alpha, const_cast<float*>(A), &lda, const_cast<float*>(B), &ldb, &beta, C, &ldc);
    return true;
}

//////////////////////////////////////////////////////////////
// wrapper for gemv call to cublas
//////////////////////////////////////////////////////////////
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
// call the host mkl back end, because the OpenMP implementation of the device back end is used
bool gemv_device(float const* A, float const* x, float* y, float alpha, float beta, int m, int n, int incx, int incy, int lda, char trans) {
    return gemv_host(A, x, y, alpha, beta, m, n, incx, incy, lda, trans);
}
bool gemv_device(double const* A, double const* x, double* y, double alpha, double beta, int m, int n, int incx, int incy, int lda, char trans) {
    return gemv_host(A, x, y, alpha, beta, m, n, incx, incy, lda, trans);
}
bool gemm_device(float const* A, float const* B, float* C, float alpha, float beta, int m, int n, int k, int lda, int ldb, int ldc, char transa, char transb) {
    return gemm_host(A,B, C, alpha, beta, m, n, k, lda, ldb, ldc, transa, transb);
}
bool gemm_device(double const* A, double const* B, double* C, double alpha, double beta, int m, int n, int k, int lda, int ldb, int ldc, char transa, char transb) {
    return gemm_host(A,B, C, alpha, beta, m, n, k, lda, ldb, ldc, transa, transb);
}
#else
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

// overloaded for double
bool gemm_device (
    double const* A, double const* B, double* C,
    double alpha, double beta,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    char transa, char transb)
{
    cublasHandle_t& handle = CublasState::instance()->handle();
    cublasOperation_t optA = transa=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t optB = transb=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasStatus_t status =
    cublasDgemm( handle, optA, optB,
                 m, n, k,
                 &alpha, A, lda, B, ldb,
                 &beta, C, ldc);
    return (status == CUBLAS_STATUS_SUCCESS);
}

// overloaded for float
bool gemm_device (
    float const* A, float const* B, float* C,
    float alpha, float beta,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    char transa, char transb)
{
    cublasHandle_t& handle = CublasState::instance()->handle();
    cublasOperation_t optA = transa=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t optB = transb=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasStatus_t status =
    cublasSgemm( handle, optA, optB,
                 m, n, k,
                 &alpha, A, lda, B, ldb,
                 &beta, C, ldc);
    return (status == CUBLAS_STATUS_SUCCESS);
}
#endif

//////////////////////////////////////////////////////////////

} //namespace detail
} //namespace threx
} //namespace minlin
