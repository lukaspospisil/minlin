/**
	Let A be block-diagonal matrix, where each block is Laplace (tridiagonal) matrix


**/

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include <iostream>

#include <stdio.h> /* printf in cuda */
#include <limits> /* max value of double/float */

#include <omp.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

using namespace minlin::threx;

/* compute on device or host ? and which tests to run ? */
#ifdef USE_GPU
	/* compute using CUDA on Device */

	#define MyVector DeviceVector
	#define MyMatrix DeviceMatrix

#else
	/* compute without CUDA on Host */

	#define MyVector HostVector
	#define MyMatrix HostMatrix

#endif

/* double/float values in Vector? */
#define Scalar double

MINLIN_INIT

/* cuda error check */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"\n\x1B[31mCUDA error:\x1B[0m %s %s \x1B[33m%d\x1B[0m\n\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* timer */
double getUnixTime(void){
	struct timespec tv;
	if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;
	return (((double) tv.tv_sec) + (double) (tv.tv_nsec / 1000000000.0));
}


class Apple {
	private:
		double value;

	public:
		void set_value(double new_value){
			this->value = new_value;
		}
		
		double get_value(){
			return this->value;
		}

		Apple operator+(const Apple &a){
			Apple a_out;
			a_out.value = this->value + a.value;
			
			return a_out;
		}
	
};

int main ( int argc, char *argv[] ) {

	std::cout << "- start of the test" << std::endl << std::endl;

	int N = 3;
	
	Apple a1, a2, a3;
	a1.set_value(5);
	a2.set_value(6);
	a3 = a1 + a2;
	
	std::cout << "test: " << a3.get_value() << std::endl << std::endl;
	
	MyVector<double> xblock(N);
	xblock(0) = 1;
	xblock(1) = 2;
	xblock(2) = 3;
	
	/* print the content of xblock */
	std::cout << "xblock:" << std::endl << xblock << std::endl << std::endl;



	MyMatrix<double> Ablock(N,N);
	
	Ablock(0,0) = 1;
	Ablock(0,1) = 2;
	Ablock(1,0) = 2;
	
	Ablock(1,1) = 3;
	Ablock(2,2) = -1;
	
	/* print the content of Ablock */
	std::cout << "Ablock:" << Ablock << std::endl;
	
	

	std::cout << "- end of the test" << std::endl;
	
	return 0;
}
