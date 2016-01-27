#--------------------------------------------------
# MinLin CMAKE sample
#
# Requirements:
# - defined variable MKLROOT
# - minlin folder with source files in CMAKE_CURRENT_SOURCE_DIR
# - 
#
#--------------------------------------------------
project(MINLIN_SAMPLE)
cmake_minimum_required(VERSION 2.8)

#load options from command line
option(USE_GPU "USE_GPU" ON)

# set default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Debug")
endif()

# set general g++ flags
# note: we always need -O3 because minlin_host doesn't compile without it
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3")

# define variables for mkl include directories
set(MKL_INCLUDE_DIR $ENV{MKLROOT}/include)
include_directories(${MKL_INCLUDE_DIR})
link_directories($ENV{MKLROOT}/lib/intel64)

# add debug definitions to compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# MINLIN: define variables for include directories
set(MINLIN_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
message(STATUS "MINLIN_INCLUDE = ${MINLIN_INCLUDE}")

if(${USE_GPU})
	message(STATUS "${Red}The code is compiled to run on GPU${ColourReset}")
	set(MINLIN_DEVICE_DEFS # we use -D here because this isn't added automatically
		-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
		-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
		-DUSE_GPU
		-DUSE_MINLIN)
else(${USE_GPU})
	message(STATUS "${Red}The code is compiled to run on CPU${ColourReset}")
	set(MINLIN_DEVICE_DEFS # we use -D here because this isn't added automatically
		-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP
		-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
		-DUSE_MINLIN)
endif(${USE_GPU})


include_directories(${MINLIN_INCLUDE})


# MINLIN DEVICE EXECUTABLE
include(FindCUDA)
set(CUDA_PROPAGATE_HOST_FLAGS off) # if flags are passed with -Xcompiler, they also affect NVCC which doesn't understand all g++ flags we use
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER}) # without this, cc is used instead of CC and all include paths have to be specified manually
string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
set(CUDA_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-vla ${CMAKE_CXX_FLAGS_${BUILD_TYPE_UPPER}}") # add flags specific to build type
string(REPLACE "-std=c++11" "" CUDA_CXX_FLAGS ${CUDA_CXX_FLAGS}) # remove C++11 from options
cuda_include_directories(${MINLIN_INCLUDE})

# test matmult
#cuda_add_executable(test_matmult test_matmult.cu
#	OPTIONS "${MINLIN_DEVICE_DEFS} -arch=sm_35 --compiler-options \"${CUDA_CXX_FLAGS}\""
#	DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
#target_link_libraries(test_matmult mkl_core mkl_gnu_thread mkl_rt boost_program_options cublas)
#set_target_properties(test_matmult PROPERTIES
#	OUTPUT_NAME test_matmult
#	)

# test sort
#cuda_add_executable(test_sort test_sort.cu
#	OPTIONS "${MINLIN_DEVICE_DEFS} -arch=sm_35 --compiler-options \"${CUDA_CXX_FLAGS}\""
#	DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
#target_link_libraries(test_sort mkl_core mkl_gnu_thread mkl_rt boost_program_options cublas)
#set_target_properties(test_sort PROPERTIES
#	OUTPUT_NAME test_sort
#	)

# test blockmatmult
#cuda_add_executable(test_blockmatmult test_blockmatmult.cu
#	OPTIONS "${MINLIN_DEVICE_DEFS} -arch=sm_35 --compiler-options \"${CUDA_CXX_FLAGS}\""
#	DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
#target_link_libraries(test_blockmatmult mkl_core mkl_gnu_thread mkl_rt boost_program_options cublas)
#set_target_properties(test_blockmatmult PROPERTIES
#	OUTPUT_NAME test_blockmatmult
#	)

# test memory
cuda_add_executable(test_memory test_memory.cu
	OPTIONS "${MINLIN_DEVICE_DEFS} -arch=sm_35 --compiler-options \"${CUDA_CXX_FLAGS}\""
	DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
target_link_libraries(test_memory mkl_core mkl_gnu_thread mkl_rt boost_program_options cublas)
set_target_properties(test_memory PROPERTIES
	OUTPUT_NAME test_memory
	)
