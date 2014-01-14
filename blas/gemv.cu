#include <iostream>

// include minlin stuff
#include <minlin/modules/threx/threx.h>
#include <minlin/minlin.h>
using namespace minlin::threx; // just dump the namespace for this example

#include <cublas_v2.h>
#include "utilities.h"

//////////////////////////////////////////////////////////////////////////////////

int main(void)
{
    typedef double real;
    using minlin::all;
    using minlin::end;
    using minlin::ColumnOriented;

    // initialize cuda stuff
    cublasHandle_t handle = init_cublas();

    // load matrices
    int m = 8;
    int n = 4;
    // allocate space for matrices and vectors, which will be used for the operation y(all)=A*x
    std::cout << "== creating m*n matrix" << std::endl;
    HostMatrix<real> A(m,n);
    std::cout << "== creating n vector" << std::endl;
    HostVector<real> x(n);
    std::cout << "== creating n vector" << std::endl;
    HostVector<real> y(m);

    std::cout << "== setting A(all) = 1" << std::endl;
    for(int i=0; i<m; i++)
        A(i,all) = real(i);
    std::cout << "== setting x(all) = 2" << std::endl;
    x(all) = 2.0;

    std::cout << "\nA\n" << A;
    std::cout << "\nx\n" << x;

    //y(all) = A*x;
    //y = A*x(0,2,n);
    y = A*x;
    std::cout << "\ny=A*x\n" << y;

    HostVector<real> z = A(m/2,all);
    std::cout << "\nz=A(1,all)\n" << z;

    // finalize
    kill_cublas(handle);

    std::cout << std::endl;
}

