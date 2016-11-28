all: lab4p1 lab4p2
lab4p1: das_piyali_lab4p1.cu
	nvcc -O -o  lab4p1  das_piyali_lab4p1.cu
lab4p2: das_piyali_lab4p2.cu
	nvcc das_piyali_lab4p2.cu bmp_reader.o  -o lab4p2
clean:
	rm -rf *.o *.out lab4p1 lab4p2
