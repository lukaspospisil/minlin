project(TEST_PETSC)
cmake_minimum_required(VERSION 2.8)

# add cmake functions
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../util/cmake/" ${CMAKE_MODULE_PATH})

#define some colors for fun
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
endif()

# set default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Debug")
endif()

# set general g++ flags
# note: we always need -O3 because minlin_host doesn't compile without it
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3")

# define variables for mkl include directories
if(NOT DEFINED ENV{MKLROOT})
  message(FATAL_ERROR "${Red}MKLROOT has to be specified!${ColourReset}")
  return()
endif()
set(MKL_INCLUDE_DIR $ENV{MKLROOT}/include)
message(STATUS "${Yellow}MKL_INCLUDE = ${MKL_INCLUDE_DIR}${ColourReset}")
include_directories(${MKL_INCLUDE_DIR})
link_directories($ENV{MKLROOT}/lib/intel64)

# add debug definitions to compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# MINLIN: define variables for include directories
set(MINLIN_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
message(STATUS "${Yellow} MINLIN_INCLUDE = ${MINLIN_INCLUDE} ${ColourReset}")
include_directories(${MINLIN_INCLUDE})


set(MINLIN_HOST_DEFS
	THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
	THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
	__host__=\ 
	__device__=\ 
	USE_MINLIN)

# MINLIN HOST EXECUTABLE
#add_executable(pascinference_host pascinference.cpp)
#target_link_libraries(pascinference_host mkl_core mkl_gnu_thread mkl_rt boost_program_options)
#set_property(
#	TARGET pascinference_host
#	APPEND
#	PROPERTY INCLUDE_DIRECTORIES
#	${MINLIN_INCLUDE})
#set_target_properties(pascinference_host PROPERTIES
#	OUTPUT_NAME pascinference_host
#	COMPILE_DEFINITIONS "${MINLIN_HOST_DEFS}"
#	)

# MINLIN DEVICE EXECUTABLE
include(FindCUDA)
set(CUDA_PROPAGATE_HOST_FLAGS off) # if flags are passed with -Xcompiler, they also affect NVCC which doesn't understand all g++ flags we use
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER}) # without this, cc is used instead of CC and all include paths have to be specified manually
string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
set(CUDA_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-vla ${CMAKE_CXX_FLAGS_${BUILD_TYPE_UPPER}}") # add flags specific to build type
string(REPLACE "-std=c++11" "" CUDA_CXX_FLAGS ${CUDA_CXX_FLAGS}) # remove C++11 from options
cuda_include_directories(${MINLIN_INCLUDE})

set(LIBRARY_DEFS "")
set(DEVICE_DEFS "")

# add USE_GPU flag to all libs and exec files

message(STATUS "${Red}The code is compiled with PETSc${ColourReset}")
set(LIBRARY_DEFS ${LIBRARY_DEFS} -DUSE_PETSC)
set(DEVICE_DEFS ${DEVICE_DEFS} -USE_PETSC)
	
# PETSc: include
if(NOT DEFINED ENV{PETSC_INCLUDES})
	message(STATUS "${Red}PETSC_INCLUDES is not specified, trying to run find_package(PETSc)${ColourReset}")
	find_package(PETSc)

	message(STATUS "${Yellow}PETSC_INCLUDES = ${PETSC_INCLUDES} ${ColourReset}")

endif()

message(STATUS "${Yellow}PETSC_DIR = $ENV{PETSC_DIR} ${ColourReset}")
message(STATUS "${Yellow}PETSC_ARCH = $ENV{PETSC_DIR} ${ColourReset}")

include_directories(${PETSC_INCLUDES})

# PETSCVECTOR: define variables for include directories
set(PETSCVECTOR_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../include/petscvector)
message(STATUS "${Yellow}PETSCVECTOR_INCLUDE = ${PETSCVECTOR_INCLUDE} ${ColourReset}")
include_directories(${PETSCVECTOR_INCLUDE})

# GENERALMATRIX: define variables for include directories
set(GENERALMATRIX_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../include/generalmatrix)
message(STATUS "${Yellow}GENERALMATRIX_INCLUDE = ${GENERALMATRIX_INCLUDE} ${ColourReset}")
include_directories(${GENERALMATRIX_INCLUDE})


set(MINLIN_DEVICE_DEFS # we use -D here because this isn't added automatically
	-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
	-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
	${LIBRARY_DEFS}
	-DUSE_MINLIN)


# -------------- MATMULT --------------

# add main executable file
cuda_add_executable(test_matmult test_matmult.cu
	OPTIONS "${MINLIN_DEVICE_DEFS} -arch=sm_35 --compiler-options \"${CUDA_CXX_FLAGS}\""
	DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
target_link_libraries(test_matmult mkl_core mkl_gnu_thread mkl_rt boost_program_options cublas ${PETSC_LIBRARIES})

# set the name of output file
set_target_properties(test_matmult PROPERTIES
	OUTPUT_NAME test_matmult
	)
	


