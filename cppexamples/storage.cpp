/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#include <thrust/functional.h>

#include <minlin/minlin.h>
#include <minlin/modules/threx/storage.h>
#include <minlin/modules/threx/operators.h>
#include <minlin/modules/threx/threx.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

//typedef thrust::host_vector<double> vec;
typedef thrust::device_vector<double> vec;

int main() {

    int N = 5;

    // allocate storage
    minlin::Vector<minlin::threx::ByValue<vec> > A(N);
    minlin::Vector<minlin::threx::ByValue<vec> > B(N);
    minlin::Vector<minlin::threx::ByValue<vec> > C(N);
    minlin::Vector<minlin::threx::ByValue<vec> > D(N);

    // initialize input vectors
    A(0) = 3;  B(0) = 6;  C(0) = 2; 
    A(1) = 4;  B(1) = 7;  C(1) = 5; 
    A(2) = 0;  B(2) = 2;  C(2) = 7; 
    A(3) = 8;  B(3) = 1;  C(3) = 4; 
    A(4) = 2;  B(4) = 8;  C(4) = 3; 

    D = A + mul(B, C);

    // print the output
    for(int i = 0; i < 5; i++)
        std::cout << A(i) << " + " << B(i) << " * " << C(i) << " = " << D(i) << std::endl;
    std::cout << cos(D) << std::endl;

    D(minlin::all) = A;
    std::cout << D << std::endl;

    D(minlin::all) = B(minlin::all);
    std::cout << D << std::endl;

    D = C(minlin::all);
    std::cout << D << std::endl;

    vec Q(5);
    Q[2] = 10;

//    minlin::Vector<minlin::threx::ByReference<vec> > QQ((minlin::threx::ByReference<vec>(Q)));
    minlin::Vector<minlin::threx::ByReference<vec> > QQ(Q); // allow this convenient syntax
    std::cout << QQ << std::endl;

    QQ(3) = 20;
    std::cout << Q[3] << std::endl; // note: Q, not QQ

    double z[] = {1, 2, 3, 8, 20};
    minlin::threx::Range<double*> Z(z, z + 5);
    minlin::Vector<minlin::threx::Range<double*> > ZZ(Z);
    std::cout << ZZ + ZZ << std::endl;

    ZZ(2) = 50;
    std::cout << ZZ << std::endl;

    std::cout << mul(ZZ, A) + B << std::endl;
    minlin::threx::Range<vec::iterator> X(Q.begin(), Q.end());
    minlin::Vector<minlin::threx::Range<vec::iterator> > XX(X);
    std::cout << XX - QQ << std::endl;

//    D = ZZ; // not allowed if D is on device device, since ZZ references host memory

    std::cout << z[0] << std::endl;
    
//    ZZ(minlin::all) = A; // not allowed if A is on device, since ZZ references host memory
    
    std::cout << z[0] << std::endl;

//    QQ = A;   // will correctly fail
    
//    ZZ = A;   // will correctly fail

}

