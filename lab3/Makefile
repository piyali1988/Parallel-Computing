all: dispopenmp persopenmp
dispopenmp: omp_disposable.cpp
	g++ -g -o dispopenmp omp_disposable.cpp -lrt -O3 -std=c++0x -fopenmp
persopenmp: omp_persistent.cpp
	g++ -g -o persopenmp omp_persistent.cpp -lrt -O3 -std=c++0x -fopenmp
clean:
	rm -rf *.o *.out dispopenmp persopenmp
