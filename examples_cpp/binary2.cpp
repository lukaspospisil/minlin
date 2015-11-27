/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#include <thrust/functional.h>

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

    std::cout << 1 + A << std::endl;
    std::cout << Vector(1 + A) << std::endl;

    std::cout << 1 - A << std::endl;
    std::cout << Vector(1 - A) << std::endl;

    std::cout << 2 * A << std::endl;
    std::cout << Vector(2 * A) << std::endl;

    std::cout << div(1, A) << std::endl;
    std::cout << Vector(div(1,  A)) << std::endl;

    std::cout << pow(2, A) << std::endl;
    std::cout << Vector(pow(2,  A)) << std::endl;

    std::cout << atan2(2, A) << std::endl;
    std::cout << Vector(atan2(2,  A)) << std::endl;

    std::cout << A + 1 << std::endl;
    std::cout << Vector(A + 1) << std::endl;

    std::cout << A - 1 << std::endl;
    std::cout << Vector(A - 1) << std::endl;

    std::cout << A * 2 << std::endl;
    std::cout << Vector(A * 2) << std::endl;

    std::cout << A / 2 << std::endl;
    std::cout << Vector(A / 2) << std::endl;

    std::cout << pow(A, 2) << std::endl;
    std::cout << Vector(pow(A, 2)) << std::endl;

    std::cout << atan2(A, 2) << std::endl;
    std::cout << Vector(atan2(A, 2)) << std::endl;

    A(minlin::all) = B + B;

}
