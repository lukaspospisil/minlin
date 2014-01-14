# greina
MKLROOT=/cm/shared/apps/intel/composer_xe/2011_sp1.12.361/mkl
CUDAROOT=/cm/shared/apps/cuda55/toolkit/5.5.22
BOOSTROOT=/users/bcumming/repos/boost_1_49_0
GPP=/cm/shared/apps/gcc/4.8.1/bin/g++

# santis
MKLROOT=/opt/intel/14.0.1.106/composer_xe_2013_sp1.1.106/mkl
CUDAROOT=/opt/nvidia/cudatoolkit/5.5.20-1.0501.7700.8.1
BOOSTROOT=/apps/santis/boost/1.54/gcc_481/include
GPP=/opt/gcc/4.8.2/bin/g++

MKLLIBS=-L$(MKLROOT)/lib/intel64 -lmkl_gnu_thread -lmkl_core -lmkl_rt
CUDALIBS=-L$(CUDAROOT)/lib64 -lcublas
LIBS=$(MKLLIBS) $(CUDALIBS)

INCLUDE=-I../include -I$(MKLROOT)/include -I$(CUDAROOT)/include -I$(BOOSTROOT)

NVCCFLAGS=-ccbin=$(GPP) --compiler-options -fopenmp

ARGS=-DUSE_GPU

all: gemv

gemv: gemv.cu utilities.h
	nvcc -arch=sm_35 gemv.cu -o gemv $(NVCCFLAGS) $(INCLUDE) $(LIBS) -DUSE_GPU -DMINLIN_DEBUG

clean :
	rm -f gemv

