all : assign assign2 binary binary2 copying dot expression index inplace logical matrix mul range range_check storage submatrix unary vector_matrix
# rand

INCLUDE=-I/scratch/santis/bcumming/minlin/release/include

assign: assign.cu
	nvcc -arch=sm_35 assign.cu -o assign $(INCLUDE)

assign2: assign2.cu
	nvcc -arch=sm_35 assign2.cu -o assign2 $(INCLUDE)

binary: binary.cu
	nvcc -arch=sm_35 binary.cu -o binary $(INCLUDE)

binary2: binary2.cu
	nvcc -arch=sm_35 binary2.cu -o binary2 $(INCLUDE)

copying: copying.cu
	nvcc -arch=sm_35 copying.cu -o copying $(INCLUDE)

dot: dot.cu
	nvcc -arch=sm_35 dot.cu -o dot $(INCLUDE)

expression: expression.cu
	nvcc -arch=sm_35 expression.cu -o expression $(INCLUDE)

index: index.cu
	nvcc -arch=sm_35 index.cu -o index $(INCLUDE)

inplace: inplace.cu
	nvcc -arch=sm_35 inplace.cu -o inplace $(INCLUDE)

logical: logical.cu
	nvcc -arch=sm_35 logical.cu -o logical $(INCLUDE)

matrix: matrix.cu
	nvcc -arch=sm_35 matrix.cu -o matrix $(INCLUDE)

mul: mul.cu
	nvcc -arch=sm_35 mul.cu -o mul $(INCLUDE)

rand: rand.cu
	nvcc -arch=sm_35 rand.cu -o rand $(INCLUDE)

range: range.cu
	nvcc -arch=sm_35 range.cu -o range $(INCLUDE)

range_check: range_check.cu
	nvcc -arch=sm_35 range_check.cu -o range_check $(INCLUDE)

storage: storage.cu
	nvcc -arch=sm_35 storage.cu -o storage $(INCLUDE)

submatrix: submatrix.cu
	nvcc -arch=sm_35 submatrix.cu -o submatrix $(INCLUDE)

unary: unary.cu
	nvcc -arch=sm_35 unary.cu -o unary $(INCLUDE)

vector_matrix: vector_matrix.cu
	nvcc -arch=sm_35 vector_matrix.cu -o vector_matrix $(INCLUDE)

clean :
	rm -f assign assign2 binary binary2 copying dot expression index inplace logical matrix mul rand range range_check storage submatrix unary vector_matrix

