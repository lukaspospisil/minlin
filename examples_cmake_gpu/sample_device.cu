/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

using namespace minlin::threx;

MINLIN_INIT

int main() {

    int N = 5;

    // allocate storage
    HostVector<double> H(N);
    DeviceVector<double> D(N);

    // initialize input vectors
    H(0) = 3;  D(0) = 6;
    H(1) = 4;  D(1) = 7;
    H(2) = 0;  D(2) = 2;
    H(3) = 8;  D(3) = 1;
    H(4) = 2;  D(4) = 8;

    std::cout << H << std::endl;
    
    std::cout << D << std::endl;
    
    D = H;
    
    std::cout << D << std::endl;

//    D(all) = range(1,5);
    
    std::cout << range(1,5) << std::endl;
    
 //   D(all) = 1;
    
    std::cout << D << std::endl;
    
}
