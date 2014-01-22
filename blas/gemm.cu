#include <iostream>

// include minlin stuff
#include <minlin/modules/threx/threx.h>
#include <minlin/minlin.h>
using namespace minlin::threx; // just dump the namespace for this example

#include <cublas_v2.h>
#include "utilities.h"

MINLIN_INIT

int main(void)
{
    typedef float real;
    using minlin::all;
    using minlin::end;
    using minlin::ColumnOriented;

    // load matrices
    int n = 1000;
    int ndisp = 10;

    // allocate space for matrices and vectors, which will be used for the operation y(all)=A*x
    HostMatrix<real> Ah(n,n);
    HostMatrix<real> Bh(n,n);

    for(int i=0; i<n; i++)
        Ah(i,i) = i+1;
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            Bh(i,j) = real(rand()%10);

    if( n<=ndisp ) {
        std::cout << "\nA\n" << Ah << std::endl;
        std::cout << "\nB\n" << Bh << std::endl;
    }
    DeviceMatrix<real> A = Ah;
    DeviceMatrix<real> B = Bh;

    DeviceMatrix<real> C;
    if(n<=ndisp)
        std::cout << "======================= performing C = A*B ==================" << std::endl;
    C = A*B;
    if(n<=ndisp)
        std::cout << "\nC=A*B\n" << C << std::endl;

    if(n<=ndisp)
        std::cout << "======================= performing C(all) = A*B ==================" << std::endl;
    C(all) = A*B;
    if(n<=ndisp)
        std::cout << "\nC(all)=A*B\n" << C << std::endl;

    if(n<=ndisp)
        std::cout << "======================= performing C = A*B(all,0,4) ==================" << std::endl;
    C = A*B(all,0,4);
    if(n<=ndisp)
        std::cout << "\nC=A*B(:,0:3)\n" << C << std::endl;

    std::cout << "======================================================================" << std::endl;

    std::cout << std::endl;
}

