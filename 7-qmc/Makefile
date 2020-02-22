NAME = complex_demo
ARCH = sm_70

CXX = nvcc
CXXFLAGS = -arch=$(ARCH) -std=c++11 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants
LDFLAGS = -lgsl -lgslcblas

.DEFAULT_GOAL := $(NAME)

.PHONY: clean
clean:
	rm -rf $(NAME)