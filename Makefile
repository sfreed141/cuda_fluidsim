CC=nvcc
CFLAGS=

TARGET=fluid
SOURCES=src/main.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm $(TARGET)
