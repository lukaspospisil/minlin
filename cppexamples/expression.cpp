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
    std::cout << B << std::endl;

    std::cout << mul(div(1 + cos(A), B) - pow(B, 1.5)/3, exp((A-1)/2), pow(div(A, B), B)) << std::endl;
}
