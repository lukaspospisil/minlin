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

    int M = 6;
    int N = 5;

    // allocate storage
    minlin::threx::DeviceMatrix<double> A(M, N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            A(i,j) = i + j/10.0;
        }
    }

    std::cout << "A(6,5)" << std::endl;
    std::cout << A << std::endl;

    std::cout << "A(2,4, 1,3)" << std::endl;
    std::cout << A(2,4, 1,3) << std::endl;

    minlin::threx::DeviceMatrix<double> B = A(2,4, 1,3);
    
    std::cout << "B" << std::endl;
    std::cout << B << std::endl;

    std::cout << A(1,4, 1,3)(1,2, 2,2) << std::endl;

    std::cout << A(minlin::all, 2,4) << std::endl;
    
    std::cout << A(0,2, minlin::all) << std::endl;
    
    std::cout << A(minlin::all, 1,4) << std::endl;

}
