CC=nvcc
CFLAGS=
LDFLAGS=
LDLIBS=-lGL -lGLU -lglut

TARGET=fluid
SOURCES=src/demo.cpp src/solver.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) 

.PHONY: clean
clean:
	rm $(TARGET)
