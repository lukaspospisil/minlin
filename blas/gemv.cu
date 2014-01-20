#include <iostream>

// include minlin stuff
#include <minlin/modules/threx/threx.h>
#include <minlin/minlin.h>
using namespace minlin::threx; // just dump the namespace for this example

//#include <cublas_v2.h>
#include "utilities.h"

//////////////////////////////////////////////////////////////////////////////////

int main(void)
{
    typedef double real;
    using minlin::all;
    using minlin::end;
    using minlin::ColumnOriented;

    // initialize cuda stuff
    //cublasHandle_t handle = init_cublas();

    // load matrices
    int m = 8;
    int n = 4;
    // allocate space for matrices and vectors, which will be used for the operation y(all)=A*x
    std::cout << "== creating m*n matrix A" << std::endl;
    HostMatrix<real> A(m,n);
    std::cout << "== creating n*n matrix B" << std::endl;
    HostMatrix<real> B(n,n);
    std::cout << "== creating m*n matrix C" << std::endl;
    HostMatrix<real> C(m,n);
    std::cout << "== creating n vector x" << std::endl;
    HostVector<real> x(n);
    std::cout << "== creating n vector y" << std::endl;
    HostVector<real> y(m);
    std::cout << "== creating 2*n vector z" << std::endl;
    HostVector<real> l(2*n);

    l(all) = 0.;
    l(0,2,2*n-1) = 1.0;

    C(all) = 0;

    std::cout << "== setting A(all,i) = i" << std::endl;
    for(int i=0; i<m; i++)
        A(i,all) = real(i);
    std::cout << "== setting B(i,all) = i-1" << std::endl;
    for(int i=0; i<n; i++)
        B(all,i) = real(i-1);
    std::cout << "== setting x(all) = 2" << std::endl;
    x(all) = 1.0;

    std::cout << "\nA\n" << A;
    std::cout << "\nB\n" << B;
    std::cout << "\nx\n" << x;

    y(all) = A*x;
    //y = A*x(0,2,n);
    //y = A*x;
    std::cout << "\ny=A*x\n" << y;

    for(int i=0; i<B.rows(); i++) {
        y = A*B(all,i);
        std::cout << "\ny=A*B(all," << i <<")\n" << y;
    }

    for(int j=0; j<C.cols(); j++) {
        C(all,j) = A*B(all,j);
    }
    std::cout << "\nC(all,j) = A*B(all,j) forall j in [0," << n << ")\n" << C;

    y = B*l(0,2,2*n-1);
    std::cout << "\ny=B*l(0,2,2*n)\n" << y;

    HostVector<real> z = A(m/2,all);
    std::cout << "\nz=A(1,all)\n" << z;

    // finalize
    //kill_cublas(handle);

    std::cout << std::endl;
}

