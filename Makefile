CC=nvcc
CFLAGS=-std=c++11 -g -O2 -lineinfo -arch=sm_52
LDFLAGS=
LDLIBS=-lGL -lGLU -lglut

TARGET=fluid
SOURCES=src/demo.cpp src/solver.cpp src/solver_cuda.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

.PHONY: clean
clean:
	rm $(TARGET)
