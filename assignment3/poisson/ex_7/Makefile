TARGET= example3d
OBJS	= main.o jacobi.o alloc3d.o alloc3d_gpu.o transfer3d_gpu.o print.o

OPT	= -g -O3
ISA	= 
PARA	= -fopenmp

XOPTS = -Xptxas=-v -arch=sm_80 -lineinfo

CC	= gcc

CCC	= g++
CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS)

CFLAGS	= $(OPT) $(ISA) $(PARA) $(XOPT)

CUDA_PATH ?= /appl/cuda/11.5.1
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

.SUFFIXES: .cu
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f *.o $(TARGET)

main.o: main.cu jacobi.h alloc3d_gpu.h alloc3d.h transfer3d_gpu.h print.h
alloc3d_gpu.o: alloc3d_gpu.cu
alloc3d.o: alloc3d.cu
transfer3d_gpu.o: transfer3d_gpu.cu
jacobi.o: jacobi.cu
print.o: print.cu