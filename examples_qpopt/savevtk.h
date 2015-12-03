#ifndef SAVEVTK_H
#define	SAVEVTK_H

#include <thrust/functional.h>

#include <iostream>
#include <fstream>

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include "minlin/vector.h"
#include "minlin/matrix.h"

using namespace minlin;



template<typename Expression>
void savevtk(const char* name_of_file, Vector<Expression> x){
	int N = x.size();
	int i;
	double h = 1.0/(N-1);

	std::ofstream myfile;
	myfile.open(name_of_file);
	myfile << "# vtk DataFile Version 3.1" << std::endl;
	myfile << "this is the solution of our problem" << std::endl;
	myfile << "ASCII" << std::endl;
	myfile << "DATASET POLYDATA" << std::endl;

	/* points - coordinates */
	myfile << "POINTS " << N << " FLOAT" << std::endl;
	for(i=0;i < N;i++){
		myfile << i*h << " " << x(i) << " 0.0" << std::endl;
	}
	
	/* line solution */
	myfile << "LINES 1 " << N+1 << std::endl;
	myfile << N << " ";
	for(i=0;i < N;i++){
		myfile << i << " ";
	}
	myfile << std::endl;
	
	/* values is points */
	myfile << "POINT_DATA " << N  << std::endl;
	myfile << "SCALARS solution float 1"  << std::endl;
	myfile << "LOOKUP_TABLE default"  << std::endl;
	for(i=0;i < N;i++){
//		myfile << x(i) << std::endl;
		myfile << 1.0 << std::endl;
	}

	myfile.close();
	
	
}

#endif
