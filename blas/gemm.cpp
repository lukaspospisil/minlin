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
    int n = 8;
    int ndisp = 10;

    // allocate space for matrices and vectors, which will be used for the operation y(all)=A*x
    HostMatrix<real> Ah(n,n);
    HostMatrix<real> Bh(n,n);

    //for(int i=0; i<n; i++)
    //    Ah(i,i) = i+1;
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            Ah(i,j) = i;
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            //Bh(i,j) = real(rand()%10);
            //Bh(i,j) = i;
            Bh(i,j) = j+i;

    if( n<=ndisp ) {
        std::cout << "\nA\n" << Ah << std::endl;
        std::cout << "\nB\n" << Bh << std::endl;
    }
    DeviceMatrix<real> A = Ah;
    DeviceMatrix<real> B = Bh;

    DeviceMatrix<real> C;
    DeviceMatrix<real> CT;
    if(n<=ndisp) std::cout << "======================= performing C = A*B ==================" << std::endl;
    C = A*B;
    if(n<=ndisp) std::cout << "\nC=A*B\n" << C << std::endl;

    if(n<=ndisp) std::cout << "======================= performing C = transpose(A)*B ==================" << std::endl;
    CT = transpose(A)*B;
    if(n<=ndisp) std::cout << "\nC=transpose(A)*B\n" << CT << std::endl;

    if(n<=ndisp) std::cout << "======================= performing C = A*transpose(B) ==================" << std::endl;
    CT = A*transpose(B);
    if(n<=ndisp) std::cout << "\nC=A*transpose(B)\n" << CT << std::endl;

    if(n<=ndisp) std::cout << "======================= performing C(all) = A*B ==================" << std::endl;
    C(all) = A*B;
    if(n<=ndisp) std::cout << "\nC(all)=A*B\n" << C << std::endl;

    if(n<=ndisp) std::cout << "======================= performing C = A*B(all,0,4) ==================" << std::endl;
    C = A*B(all,3,6);
    if(n<=ndisp) std::cout << "\nC=A*B(:,3:6)\n" << C << std::endl;

    DeviceMatrix<real> D = B(all,0,2);
    if(n<=ndisp) std::cout << "\nD=B(:,0:2)\n" << D << std::endl;


    std::cout << "======================================================================" << std::endl;

    std::cout << std::endl;
}

