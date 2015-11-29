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

all : assign assign2 binary binary2 copying dot expression index inplace logical matrix matmult mul range range_check storage submatrix unary vector_matrix
# rand

CC=g++ -fopenmp

MINLINROOT=/home/lukas/soft/minlin
THRUSTROOT=/home/lukas/soft
MKLROOT=/opt/intel/compilers_and_libraries_2016.1.150/linux
#MKLROOT=/opt/intel/compilers_and_libraries_2016.0.109/linux

INCLUDE= -I$(MINLINROOT) -I$(MINLINROOT)/include
INCLUDE+= -I$(MKLROOT)/mkl/include
INCLUDE+= -I$(THRUSTROOT)
LIBS+= -L$(MKLROOT)/mkl/lib/intel64_lin -lmkl_core -lmkl_rt -lmkl_gnu_thread
LIBS+= -L$(MKLROOT)/compiler/lib/intel64_lin -liomp5


# this runs with only openmpi
INCLUDE+=-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP

assign: assign.cpp
	$(CC) $(INCLUDE) assign.cpp -o assign $(LIBS) 

assign2: assign2.cpp
	$(CC) $(INCLUDE) assign2.cpp -o assign2 $(LIBS)

binary: binary.cpp
	$(CC) $(INCLUDE) binary.cpp -o binary $(LIBS) 

binary2: binary2.cpp
	$(CC) $(INCLUDE) binary2.cpp -o binary2 $(LIBS) 

copying: copying.cpp
	$(CC) $(INCLUDE) copying.cpp -o copying $(LIBS) 

dot: dot.cpp
	$(CC) $(INCLUDE) dot.cpp -o dot $(LIBS) 

expression: expression.cpp
	$(CC) $(INCLUDE) expression.cpp -o expression $(LIBS) 

index: index.cpp
	$(CC) $(INCLUDE) index.cpp -o index $(LIBS) 

inplace: inplace.cpp
	$(CC) $(INCLUDE) inplace.cpp -o inplace $(LIBS) 

logical: logical.cpp
	$(CC) $(INCLUDE) logical.cpp -o logical $(LIBS) 

matrix: matrix.cpp
	$(CC) $(INCLUDE) matrix.cpp -o matrix $(LIBS) 

matmult: matmult.cpp
	$(CC) $(INCLUDE) matmult.cpp -o matmult $(LIBS) 

mul: mul.cpp
	$(CC) $(INCLUDE) mul.cpp -o mul $(LIBS) 

rand: rand.cpp
	$(CC) $(INCLUDE) rand.cpp -o rand $(LIBS) 

range: range.cpp
	$(CC) $(INCLUDE) range.cpp -o range $(LIBS) 

range_check: range_check.cpp
	$(CC) $(INCLUDE) range_check.cpp -o range_check $(LIBS) 

storage: storage.cpp
	$(CC) $(INCLUDE) storage.cpp -o storage $(LIBS) 

submatrix: submatrix.cpp
	$(CC) $(INCLUDE) submatrix.cpp -o submatrix $(LIBS) 

unary: unary.cpp
	$(CC) $(INCLUDE) unary.cpp -o unary $(LIBS) 

vector_matrix: vector_matrix.cpp
	$(CC) $(INCLUDE) vector_matrix.cpp -o vector_matrix $(LIBS) 

clean :
	rm -f assign assign2 binary binary2 copying dot expression index inplace logical matrix matmult mul rand range range_check storage submatrix unary vector_matrix

