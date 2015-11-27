/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

using namespace minlin::threx;

#include <iostream>

int main() {

    int N = 5;

    // allocate storage
    DeviceVector<double> A(N);
    DeviceVector<double> B(N);

    // initialize input vectors
    A(0) = 3;  B(0) = 6;
    A(1) = 4;  B(1) = 7;
    A(2) = 0;  B(2) = 2;
    A(3) = 8;  B(3) = 1;
    A(4) = 2;  B(4) = 8;

    std::cout << A << std::endl;
    
    std::cout << B << std::endl;

    std::cout << mul(A, B) << std::endl;

    std::cout << DeviceVector<double>(mul(A, B)) << std::endl;

    std::cout << mul(A, B, A) << std::endl;

    std::cout << DeviceVector<double>(mul(A, B, A)) << std::endl;

    std::cout << mul(A, B, A, B) << std::endl;
    
    std::cout << DeviceVector<double>(mul(A, B, A, B)) << std::endl;

    std::cout << mul(A, B, A, B, B) << std::endl;
    
    std::cout << DeviceVector<double>(mul(A, B, A, B, B)) << std::endl;

    std::cout << mul(A, B, A, B, B, A) << std::endl;

    std::cout << DeviceVector<double>(mul(A, B, A, B, B, A)) << std::endl;

    std::cout << mul(A, B, A, B, B, A, A) << std::endl;

    std::cout << DeviceVector<double>(mul(A, B, A, B, B, A, A)) << std::endl;

    std::cout << mul(A, B, A, B, B, A, A, A) << std::endl;

    std::cout << DeviceVector<double>(mul(A, B, A, B, B, A, A, A)) << std::endl;

    std::cout << mul(A, B, A, B, B, A, A, A, B) << std::endl;

    std::cout << DeviceVector<double>(mul(A, B, A, B, B, A, A, A, B)) << std::endl;

    std::cout << mul(A, B, A, B, B, A, A, A, B, A) << std::endl;

    std::cout << DeviceVector<double>(mul(A, B, A, B, B, A, A, A, B, A(1,2))) << std::endl;
    
}

