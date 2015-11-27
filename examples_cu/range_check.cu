/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#define MINLIN_DEBUG 1

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
    A(1) = 4;  B(1) = 7;  C(1) = 5; 
    A(2) = 0;  B(2) = 2;  C(2) = 7; 
    A(3) = 8;  B(3) = 1;  C(3) = 4; 
    A(4) = 2;  B(4) = 8;  C(4) = 3; 

	minlin::threx::DeviceVector<int> I(N);
	I(0) = 4; I(1) = 3; I(2) = 2; I(3) = 1; I(4) = 0;

	minlin::threx::DeviceVector<int> J(N-1);
	J(0) = 4; J(1) = 3; J(2) = 2; J(3) = 1;

	std::cout << A(I) << std::endl;

	std::cout << B << std::endl;

	std::cout << B(minlin::all) << std::endl;
	
//	std::cout << B(-1) << std::endl;
	
//	std::cout << B(5) << std::endl;

	std::cout << B(I) << std::endl;
	
//	I(2) = -1;
	
//	std::cout << B(I) << std::endl;

	A = B;
	
	std::cout << A << std::endl;
	
	A(minlin::all) = B;

	std::cout << A << std::endl;	

	A = B(J);

	std::cout << A << std::endl;

	C(minlin::all) = B(I);
	
	std::cout << C << std::endl;
	
//	C += B(J);
}
