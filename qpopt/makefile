# I assume that MKL is installed and added to LD_LIBRARY_PATH
# in ubuntu: 
# - create /etc/ld.so.conf.d/mkl.conf and add path to mkl libraries 
# /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64_lin/
# /opt/intel/compilers_and_libraries_2016.0.109/linux/compiler/lib/intel64_lin
# - run sudo ldconfig
#
# I also need Thrust
#
# and boost: sudo apt-get install libboost-dev
#

all : qpopt
# rand

CC=g++ -fopenmp
GCC=gcc -shared

MINLINROOT=/home/lukas/soft/minlin
THRUSTROOT=/home/lukas/soft
MKLROOT=/opt/intel/compilers_and_libraries_2016.0.109/linux

INCLUDE= -I$(MINLINROOT) -I$(MINLINROOT)/include
INCLUDE+= -I$(MKLROOT)/mkl/include
INCLUDE+= -I$(THRUSTROOT)
INCLUDE+= -Iinclude/
LIBS+= -L$(MKLROOT)/mkl/lib/intel64_lin -lmkl_core -lmkl_rt -lmkl_gnu_thread
LIBS+= -L$(MKLROOT)/compiler/lib/intel64_lin -liomp5

# this runs with only openmpi
INCLUDE+=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP

qpopt: qpopt.cpp
	$(CC) -c $(INCLUDE) qpopt.cpp -o qpopt.o $(LIBS) -fpic
	$(CC) -o lib/libqpopt.so qpopt.o $(LIBS)
	rm -f qpopt.o

clean :
	rm -f qpopt.o lib/libqpopt.su

