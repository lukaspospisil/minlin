/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

//typedef minlin::threx::HostVector<double> Vector;
typedef minlin::threx::DeviceVector<double> Vector;

int main() {

    int N = 5;

    Vector A(N);
    Vector B(N);

    // initialize input vectors
    A(0) = 3;  B(0) = 6;
    A(1) = 4;  B(1) = 7;
    A(2) = 0;  B(2) = 2;
    A(3) = 8;  B(3) = 1;
    A(4) = 2;  B(4) = 8;

    std::cout << A << std::endl;
    std::cout << B << std::endl;

    A += B;
    std::cout << A << std::endl;

    A -= B;
    std::cout << A << std::endl;

    A *= B(minlin::all);
    std::cout << A << std::endl;

    A(minlin::all) /= B;
    std::cout << A << std::endl;

    A(minlin::all) = 1;
    std::cout << A << std::endl;

    B += 1;
    std::cout << B << std::endl;

    B -= 1;
    std::cout << B << std::endl;

    B *= 2;
    std::cout << B << std::endl;

    B(minlin::all) /= 2;
    std::cout << B << std::endl;

}
