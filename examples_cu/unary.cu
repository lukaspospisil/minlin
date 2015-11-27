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

    std::cout << +H << std::endl;

    std::cout << -D << std::endl;

    std::cout << abs(cos(H)) << std::endl;
    
    std::cout << DeviceVector<double>(cos(D)) << std::endl;
    
    std::cout << DeviceVector<double>(abs(cos(D))) << std::endl;

    std::cout << abs(DeviceVector<double>(cos(D))) << std::endl;
    
    std::cout << exp(sin(cos(abs(sqrt(log(cosh(tan(D)))))))) << std::endl;
    
}

