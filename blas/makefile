# daint
#MKLROOT=/opt/intel/13.1.0.146/composer_xe_2013.2.146/mkl
#CUDAROOT=/opt/nvidia/cudatoolkit/5.5.20-1.0501.7700.8.1
#BOOSTROOT=/apps/santis/boost/1.54/gcc_481/include
#GPP=/opt/gcc/4.8.1/bin/g++

# greina
#MKLROOT=/cm/shared/apps/intel/composer_xe/2011_sp1.12.361/mkl
#CUDAROOT=/cm/shared/apps/cuda55/toolkit/5.5.22
#BOOSTROOT=/users/bcumming/repos/boost_1_49_0
#GPP=/cm/shared/apps/gcc/4.7.1/bin/g++

# santis
#MKLROOT=/opt/intel/14.0.1.106/composer_xe_2013_sp1.1.106/mkl
#CUDAROOT=/opt/nvidia/cudatoolkit/5.5.20-1.0501.7700.8.1
#BOOSTROOT=/apps/santis/boost/1.54/gcc_481/include
#GPP=/opt/gcc/4.8.2/bin/g++

# Lukas
MKLROOT=/opt/intel/mkl
CUDAROOT=/usr/local/cuda-6.5/
BOOSTROOT=/usr/include/boost/
GPP=gcc

MKLLIBS=-L$(MKLROOT)/lib/intel64/ -lmkl_gnu_thread -lmkl_core -lmkl_rt 
CUDALIBS=-L$(CUDAROOT)/lib64 -lcudart -lcublas -lcusparse
LIBS=$(MKLLIBS) $(CUDALIBS)

INCLUDE=-I../include -I$(MKLROOT)/include -I$(CUDAROOT)/include -I$(BOOSTROOT) -I/usr/include/openmpi-x86_64/ 

NVCCFLAGS=-ccbin=$(GPP) --compiler-options -fopenmp

ARGS=-DUSE_GPU -DMINLIN_DEBUG
ARGS=-DUSE_GPU -g

all: gemv gemm

gemv: gemv.cu gemv.cpp utilities.h
#	nvcc -arch=sm_35 gemv.cu -o gemv $(NVCCFLAGS) $(INCLUDE) $(LIBS) $(ARGS)
	nvcc -arch=sm_35 gemv.cu -c -o gemv.o $(NVCCFLAGS) $(INCLUDE) $(LIBS) $(ARGS) -Xcompiler -fpic
	mpicxx -v gemv.o -o gemv $(INCLUDE) $(LIBS) $(ARGS) -fopenmp
	gcc -shared -o libgemv.so gemv.o

gemm: gemm.cu gemm.cpp utilities.h
	nvcc -arch=sm_35 gemm.cu -c -o gemm.o $(NVCCFLAGS) $(INCLUDE) $(LIBS) $(ARGS) -Xcompiler -fpic
	mpicxx -v gemm.o -o gemm $(INCLUDE) $(LIBS) $(ARGS) -fopenmp
	gcc -shared -o libgemm.so gemm.o

clean :
	rm -f gemv gemm gemv.o gemm.o libgemm.so libgemv.so

