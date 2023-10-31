NAME = vectorsum
CUDA_ARCH = -gencode arch=compute_80,code=sm_80

CXX = nvcc
CXXFLAGS = $(CUDA_ARCH) -std=c++14 -rdc=true -x cu

.DEFAULT_GOAL := $(NAME)

.PHONY: clean
clean:
	rm -rf $(NAME)
