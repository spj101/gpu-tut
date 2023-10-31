NAME = complex_demo
ARCH = sm_70

CXX = nvcc
CXXFLAGS = -arch=$(ARCH) -std=c++14 -rdc=true -x cu -Xptxas -O0 -Xptxas --disable-optimizer-constants `gsl-config --cflags`
LDFLAGS = `gsl-config --libs`

.DEFAULT_GOAL := $(NAME)

.PHONY: clean
clean:
	rm -rf $(NAME)
