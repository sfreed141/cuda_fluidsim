CC=nvcc
CFLAGS=
LDFLAGS=
LDLIBS=-lGL -lGLU -lglut

TARGET=fluid
SOURCES=src/demo.c src/solver.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS) 

.PHONY: clean
clean:
	rm $(TARGET)
