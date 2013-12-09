/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

int main() {

    int N = 5;

    // allocate storage
    minlin::threx::DeviceVector<double> A(N);
    minlin::threx::DeviceVector<double> B(N);
    minlin::threx::DeviceVector<double> C(N);
    minlin::threx::DeviceVector<double> D(N);

    // initialize input vectors
    A(0) = 3;  B(0) = 6;  C(0) = 2; 
    A(1) = 4;  B(1) = 7;  C(1) = 4; 
    A(2) = 0;  B(2) = 2;  C(2) = 7; 
    A(3) = 8;  B(3) = 1;  C(3) = 4; 
    A(4) = 2;  B(4) = 8;  C(4) = 2; 

    std::cout << (A == C) << std::endl;

    std::cout << (4 == C) << std::endl;
    
    std::cout << any_of(4 == C) << std::endl;
    
    std::cout << any_of(A == B) << std::endl;

    std::cout << any_of(A) << std::endl;    

    std::cout << (B == 7) << std::endl;

    std::cout << minlin::threx::DeviceVector<double>(B == 7) << std::endl;

    std::cout << minlin::threx::DeviceVector<double>(7 <= B) << std::endl;

    std::cout << (A != B) << std::endl;

    std::cout << minlin::threx::DeviceVector<double>(A != B) << std::endl;

    std::cout << ((A == B) || (A == C)) << std::endl;

    std::cout << minlin::threx::DeviceVector<double>(!A) << std::endl;
}
