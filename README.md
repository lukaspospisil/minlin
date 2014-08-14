minlin
======

Minimal linear algebra library. A wrapper around the Thrust template library for CUDA and OpenMP, that uses expression templates to mimic, as closely as C++ will allow, the syntax of Matlab (while avoiding much of the tempory memory allocation performed by Matlab.

See the examples path for a how-to by example.

Vector-vector operations are very well supported, along with per-elemnt matrix operations.

On going development work is working towards efficient implementations of Level-2 and Level-3 BLAS routines (such as gemv and gemm) with optimised BLAS libraries such as MKL on the host, and CUBLAS on the device.

Originally developed by Timothy Moroney @ QUT, Brisbane, Australia.

Currently maintained by Ben Cumming @ CSCS, Lugano, Switzerland. All communication about the library should be directed to Ben.

See the LICENSE file for license information.
