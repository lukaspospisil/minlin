# I assume that MKL is installed and added to LD_LIBRARY_PATH
# in ubuntu: 
# - create /etc/ld.so.conf.d/mkl.conf and add path to mkl libraries /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64_lin/
# run sudo ldconfig
#
# I also need Thrust
#
# and boost: sudo apt-get install libboost-dev
#

all : qpopt
# rand

CC=g++ -fopenmp

MINLINROOT=/home/lukas/soft/minlin
THRUSTROOT=/home/lukas/soft
MKLROOT=/opt/intel/compilers_and_libraries/linux/mkl

INCLUDE= -I$(MINLINROOT) -I$(MINLINROOT)/include
INCLUDE+= -I$(MKLROOT)/include/
INCLUDE+= -I$(THRUSTROOT)
LIBS+= -L$(MKLROOT)/lib/intel64_lin/ -lmkl_core -lmkl_rt -lmkl_gnu_thread -liomp5

# this runs with only openmpi
INCLUDE+=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP

qpopt: qpopt.cpp
	$(CC) $(INCLUDE) assign.cpp -o assign $(LIBS) 

clean :
	rm -f assign assign2 binary binary2 copying dot expression index inplace logical matrix matmult mul rand range range_check storage submatrix unary vector_matrix

