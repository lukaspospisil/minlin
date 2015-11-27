/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#include <thrust/functional.h>

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

int main() {

    int N = 5;

    // allocate storage
    minlin::threx::HostVector<double> A(N);
    minlin::threx::HostVector<double> B(N);

    // initialize input vectors
    A(0) = 3;  B(0) = 6;
    A(1) = 4;  B(1) = 7;
    A(2) = 0;  B(2) = 2;
    A(3) = 8;  B(3) = 1;
    A(4) = 2;  B(4) = 8;

    std::cout << A << std::endl;
    
    std::cout << B << std::endl;

    std::cout << minlin::threx::repvec(10.0, 8) << std::endl;
   
    A = minlin::threx::repvec(2, 8);
    
    B = minlin::threx::repvec(-2, 8);
    
    std::cout << A << std::endl;
    std::cout << B << std::endl;
    
    A = minlin::threx::zeros(5);
    B = minlin::threx::ones(5);
    
    std::cout << A << std::endl;
    std::cout << B << std::endl;

    B = minlin::threx::range(-5, -1);
    
    std::cout << B << std::endl;

    std::cout << minlin::threx::range(0, 2, 5) << std::endl;

    std::cout << minlin::threx::range(0, -2, 5) << std::endl;

    std::cout << minlin::threx::range(5, -2, 0) << std::endl;

    std::cout << minlin::threx::range(-5, -2, 0) << std::endl;

    std::cout << minlin::threx::range(0, -2, -5) << std::endl;

    std::cout << minlin::threx::range(1.0, -0.1, 0.0) << std::endl;

}

