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
    minlin::threx::DeviceVector<double> D(N);
    minlin::threx::DeviceVector<double> C(N);

    // initialize input vectors
    C(0) = 3;  D(0) = 6;
    C(1) = 4;  D(1) = 7;
    C(2) = 0;  D(2) = 2;
    C(3) = 8;  D(3) = 1;
    C(4) = 2;  D(4) = 8;

    std::cout << C << std::endl;
    
    std::cout << D << std::endl;
    
    C(minlin::all) = D;
    
    std::cout << C << std::endl;

    C(minlin::all) += minlin::threx::ones(5);

    std::cout << C << std::endl;

    C -= minlin::threx::ones(5);

    std::cout << C << std::endl;

}
