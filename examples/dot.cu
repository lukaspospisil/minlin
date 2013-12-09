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

    DeviceVector<int> I = range(1, 5);

    // initialize input vectors
    H(0) = 3;  D(0) = 6;
    H(1) = 4;  D(1) = 7;
    H(2) = 0;  D(2) = 2;
    H(3) = 8;  D(3) = 1;
    H(4) = 2;  D(4) = 8;

    std::cout << H << std::endl;
    
    std::cout << D << std::endl;

	std::cout << dot(H, H) << std::endl;
	
	std::cout << dot(D, D) << std::endl;

	std::cout << norm(D) << std::endl;

    std::cout << dot(I, I) << std::endl;

    std::cout << norm(D, 1) << std::endl;
    
    std::cout << norm(D, 2) << std::endl;

    std::cout << norm(D, inf) << std::endl;

    std::cout << norm(D, -inf) << std::endl;

    std::cout << norm(D, 3) << std::endl;
    
    std::cout << length(D) << std::endl;
    
}

