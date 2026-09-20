NVCC ?= nvcc
TARGET = two_gpu_connect4

all: $(TARGET)

$(TARGET): src/two_gpu_connect4.cu
	$(NVCC) -std=c++17 -O2 -Xcompiler -pthread $< -o $@

clean:
	rm -f $(TARGET)
