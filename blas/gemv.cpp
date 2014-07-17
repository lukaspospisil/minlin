#include <iostream>

#include <mpi.h>

// include minlin stuff
#include <minlin/modules/threx/threx.h>
#include <minlin/minlin.h>
using namespace minlin::threx; // just dump the namespace for this example

#include <cublas_v2.h>
#include "utilities.h"

MINLIN_INIT

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    typedef double real;
    using minlin::all;
    using minlin::end;
    using minlin::ColumnOriented;

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

    //// copy host matrices to the device
    DeviceMatrix<real> Ad = A;
    DeviceMatrix<real> Bd = B;
    DeviceVector<real> yd = y;

    std::cout << "\n================= non-transpose gemv =================" << std::endl;
    y(all) = A*x;
    std::cout << "\ny=A*x\n" << y << std::endl;

    for(int i=0; i<B.cols(); i++) {
        y(all) = A*B(all,i);
        std::cout << "host   : y=A*B(all," << i <<")\n" << y << std::endl;
    }

    for(int i=0; i<B.cols(); i++) {
        yd(all) = Ad*Bd(all,i);
        std::cout << "device : y=A*B(all," << i <<")\n" << yd << std::endl;
    }

    for(int j=0; j<C.cols(); j++) {
        C(all,j) = A*B(all,j);
    }
    std::cout << "\nC(all,j) = A*B(all,j) forall j in [0," << n << ")\n" << C;

    y = B*l(0,2,2*n-1);
    std::cout << "\ny=B*l(0,2,2*n)\n" << y;

    HostVector<real> z = A(m/2,all);
    std::cout << "\nz=A(1,all)\n" << z;

    std::cout << "\n================= transpose gemv =================" << std::endl;

    DeviceVector<real> xt(8);
    xt(all) = 1.0;
    std::cout << "\nxt=\n" << xt << std::endl;

    DeviceVector<real> yt(8);
    yt(0,3) = transpose(Ad)*xt;
    std::cout << yt.pointer() << " = " << Ad.pointer() << " * " << xt.pointer() << std::endl;
    std::cout << "\ny=transpose(A)*x\n" << yt << std::endl;

    MPI_Finalize();
}

