/*******************************************************************************
   minlin library.
   copyright 2013 Timothy Moroney: t.moroney@qut.edu.au
   licensed under BSD license (see LICENSE.txt for details)
*******************************************************************************/
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

int main() {

    int M = 3;
    int N = 5;

    minlin::threx::DeviceMatrix<double> A(M, N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            A(i,j) = i + j/10.0;
        }
    }

    std::cout << A << std::endl;

    minlin::threx::DeviceVector<double> row = A(0, minlin::all);
    minlin::threx::DeviceVector<double> col = A(minlin::all, 1);

    std::cout << row << std::endl;
    std::cout << col << std::endl;

    minlin::threx::DeviceMatrix<double> Row(1, N);
    minlin::threx::DeviceMatrix<double> Col(M, 1);

    row = Row;
    col = Col;

    row(minlin::all) = Row;
    col(0,2) = Row(0,2);

    minlin::threx::DeviceMatrix<double> B = row;
    B = col;

    col = B;
}
