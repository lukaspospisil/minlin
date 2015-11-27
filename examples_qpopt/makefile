# Lukas Pospisil, lukas.pospisil@usi.ch, 2015
#
# I assume that MKL is installed and added to LD_LIBRARY_PATH
# in ubuntu: 
# - create /etc/ld.so.conf.d/mkl.conf and add path to mkl libraries /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64_lin/
# run sudo ldconfig
#
# I also need Thrust
#
# and boost: sudo apt-get install libboost-dev
#

all : string

CC=g++ -fopenmp

MINLINROOT=/home/lukas/soft/minlin
THRUSTROOT=/home/lukas/soft
MKLROOT=/opt/intel/compilers_and_libraries_2016.0.109/linux

INCLUDE= -I$(MINLINROOT) -I$(MINLINROOT)/include
INCLUDE+= -I$(MKLROOT)/mkl/include
INCLUDE+= -I$(THRUSTROOT)
LIBS+= -L$(MKLROOT)/mkl/lib/intel64_lin -lmkl_core -lmkl_rt -lmkl_gnu_thread
LIBS+= -L$(MKLROOT)/compiler/lib/intel64_lin -liomp5

# include qpopt
QPOPTROOT=$(MINLINROOT)/qpopt
INCLUDE= -I$(QPOPTROOT)/include
LIBS+= -L$(QPOPTROOT)/lib -lqpopt

# this runs with only openmpi
INCLUDE+=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP

string: string.cpp
	$(CC) $(INCLUDE) string.cpp -o string $(LIBS) 

clean :
	rm -f string

