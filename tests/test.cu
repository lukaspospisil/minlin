/**
	general test, here could be anything

**/

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

#include <stdio.h> /* printf in cuda */
#include <stdlib.h> /* atoi, strtol */
#include <limits> /* max value of double/float */

#include <omp.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <typeinfo>

using namespace minlin::threx;

MINLIN_INIT

void fun_with_all(minlin::detail::all_type add_in){
	std::cout << "it works!" << std::endl;
}


int main ( int argc, char *argv[] ) {

	/* what the hell is "all"? */
	std::cout << "type of all: ";
	std::cout << typeid(all).name();
	std::cout << std::endl;
	
	fun_with_all(all);
}
