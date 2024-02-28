CC      := /usr/bin/g++
CUDA_PATH ?= "/usr/local/cuda-9.0"
NVCC    := $(CUDA_PATH)/bin/nvcc -ccbin $(CC)
#NVCC :=nvcc
# Gencode arguments
#SMS ?= 35 37 50 52 60 61 70 75
#$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
#HIGHEST_SM := $(lastword $(sort $(SMS)))
#ifneq ($(HIGHEST_SM),)
#GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
#endif
#endif


#CUDA_LDFLAGS := -lrt -lcudart -m64 -lcuda -std=c++11 -g -rdc=true
#CUDA_LDFLAGS := 
CUDA_ARCH := -arch=sm_35

#all: dynamicparallelism_Memcpy
all: dynamicparallelism_MallocManaged
#	dynamicparallelism_variable dynamicparallelism_MallocManaged

dynamicparallelism.o: dynamicparallelism.cu
	$(NVCC) ${CUDA_ARCH} -dc $< -o $@ 
	#$(NVCC) -arch=sm_35 -dc dynamicparallelism.cu -o dynamicparallelism.o 

dynamicparallelism: dynamicparallelism.o
	$(NVCC) ${CUDA_ARCH} $< -lcudadevrt -o $@
#	$(NVCC) -arch=sm_35 dynamicparallelism.o -lcudadevrt -o dynamicparallelism
#
dynamicparallelism_variable.o: dynamicparallelism_variable.cu
	$(NVCC) ${CUDA_ARCH} -dc $< -o $@ 
	#$(NVCC) -arch=sm_35 -dc dynamicparallelism.cu -o dynamicparallelism.o 

dynamicparallelism_variable: dynamicparallelism_variable.o
	$(NVCC) ${CUDA_ARCH} $< -lcudadevrt -lcudart -lcuda -o $@
	rm -rf *.o

dynamicparallelism_MallocManaged.o: dynamicparallelism_MallocManaged.cu
	$(NVCC) ${CUDA_ARCH} -dc $< -o $@ 
	#$(NVCC) -arch=sm_35 -dc dynamicparallelism.cu -o dynamicparallelism.o 

dynamicparallelism_MallocManaged: dynamicparallelism_MallocManaged.o
	$(NVCC) ${CUDA_ARCH} $< -lcudadevrt -lcuda -o $@
	rm -rf *.o

dynamicparallelism_Memcpy.o: dynamicparallelism_Memcpy.cu
	$(NVCC) ${CUDA_ARCH} -dc $< -o $@ 
	
dynamicparallelism_Memcpy: dynamicparallelism_Memcpy.o
	$(NVCC) ${CUDA_ARCH} $< -lcudadevrt -lcuda -o $@
	rm -rf *.o


clean: 
	rm -rf *.o
	rm -rf dynamicparallelism
	rm -rf dynamicparallelism_function
	rm -rf dynamicparallelism_MallocManaged
	rm -rf dynamicparallelism_Memcpy

