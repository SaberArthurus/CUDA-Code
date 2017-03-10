LDFLAGS +=  $(shell pkg-config --libs opencv)
CXXFLAGS +=  $(shell pkg-config --cflags opencv)
main: main.cu aux.cu aux.h
	nvcc main.cu aux.cu  -o main $(LDFLAGS) $(CXXFLAGS)

.PHONY: clean
clean:
	-rm -rf main