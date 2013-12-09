/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

using namespace minlin::threx;

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

    HostVector<double> H2 = D;

    std::cout << H2 << std::endl;

    DeviceVector<double> D2 = H;
    
    std::cout << D2 << std::endl;

    DeviceVector<double> D3 = mul(D + D2, D2 + D) - D2;

    HostVector<double> H3 = mul(H + H2, H2 + H) - H2;
    
    std::cout << H3 << std::endl;
    
    HostVector<double> Hlong(10);
    
    H3 = Hlong;
    
    std::cout << H3 << std::endl;
    
    D3 = Hlong;
    
    std::cout << D3 << std::endl;

    DeviceVector<double> Dlong(10);
    Dlong(1) = 10;
    
    D3 = Dlong;
    
    std::cout << D3 << std::endl;
    
    H3 = Dlong;
    
    std::cout << H3 << std::endl;
    
    H3 = H + H2;
    
    std::cout << H3 << std::endl;

    D3 = D - D2;
    
    std::cout << D3 << std::endl;

    D3(all) = D2;
    
    std::cout << D2 << std::endl;
    
}
