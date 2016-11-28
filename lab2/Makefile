all: disposable persistent
disposable: das_piyali_disposable.cpp
	g++ -g -o disposable das_piyali_disposable.cpp -lpthread -lrt -O3 -std=c++0x
persistent: das_piyali_persistent.cpp
	g++ -g -o persistent das_piyali_persistent.cpp -lpthread -lrt -O3 -std=c++0x
clean:
	rm -rf *.o *.out disposable persistent
