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

    int N = 3;

    // allocate storage
    DeviceMatrix<double> A(N, N);
    DeviceVector<double> x(N);
    DeviceVector<double> b(N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A(i,j) = N*i + j;
        }
        
        x(i) = -1.0-i;
    }
    
    std::cout << "A:" << std::endl;
    std::cout << A << std::endl << std::endl;

    std::cout << "x:" << std::endl;
    std::cout << x << std::endl << std::endl;

	b = A*x;
    
    std::cout << "b:" << std::endl;
    std::cout << b << std::endl << std::endl;


}
