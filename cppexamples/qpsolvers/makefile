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

DIR_MINLIN=/home/lukas/soft/minlin
DIR_THRUST=/home/lukas/soft
DIR_MKL=/opt/intel/compilers_and_libraries/linux/mkl

INCLUDE= -I$(DIR_MINLIN)/include
INCLUDE+= -I$(DIR_MINLIN)
INCLUDE+= -I$(DIR_MKL)/include/
INCLUDE+= -I$(DIR_THRUST)
LIBS+= -L$(DIR_MKL)/lib/intel64_lin/ -lmkl_core -lmkl_rt -lmkl_gnu_thread -liomp5

# this runs with only openmpi
INCLUDE+=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP

string: string.cpp
	$(CC) $(INCLUDE) string.cpp -o string $(LIBS) 

clean :
	rm -f string

