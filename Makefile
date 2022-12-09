
NVCC_OPTIONS = -g -std=c++14

all: lab3_ex1 lab3_ex2 lab3_ex3

lab3_ex1: lab3_ex1.cu
	nvcc $(NVCC_OPTIONS) $< -o $@

lab3_ex2: lab3_ex2.cu
	nvcc $(NVCC_OPTIONS) $< -o $@

lab3_ex3: lab3_ex3.cu
	nvcc $(NVCC_OPTIONS) $< -o $@

clean:
	rm lab3_ex1
	rm lab3_ex2
	rm lab3_ex3
.PHONY: clean