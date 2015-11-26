/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#include <thrust/functional.h>

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

using namespace minlin::threx;

int main() {

    int M = 3;
    int N = 5;

    // allocate storage
    DeviceMatrix<double> A(M, N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            A(i,j) = i + j/10.0;
        }
    }
    
    std::cout << A << std::endl;
    
    DeviceMatrix<double> B = exp(A);
    
    std::cout << B << std::endl;
    
    std::cout << acos(cos(A))(minlin::all) << std::endl;

    A(minlin::all, 1) = 9.9;
    
    std::cout << A << std::endl;

    std::cout << A(0, minlin::all) << std::endl;
    std::cout << A(1, minlin::all) << std::endl;
    std::cout << A(2, minlin::all) << std::endl;
    
    A(1, minlin::all) = -9.9;
    std::cout << A << std::endl;
    
    std::cout << A(2, minlin::all)(0, 2, 5) << std::endl;

    std::cout << mul(A, -A) - mul(-A, A) << std::endl;

    std::cout << "div(mul(A, 2*A, 3*A), mul(A, A, 6*A))"  << std::endl;
    std::cout << div(mul(A, 2*A, 3*A), mul(A, A, 6*A)) << std::endl;

    std::cout << (A == 2*A-2) << std::endl;
    
    std::cout << any_of(A == 2*A-2) << std::endl;

    A = 2*A;

    A(minlin::all) = A / 2;

    std::cout << A << std::endl;
}
