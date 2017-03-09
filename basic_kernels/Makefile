CXX := nvcc

all: addArrays

addArrays: addArrays.cu
	$(CXX) $< -o $@
.PHONY: clean

clean:
	-rm -f addArrays