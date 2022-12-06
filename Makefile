
all: lab3_ex1

lab3_ex1: lab3_ex1.cu
	nvcc -g $< -o $@

clean:
	rm lab3_ex1
.PHONY: clean