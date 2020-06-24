all : test_adaline interpolator

test_adaline : main.cpp adaline.o adaline_network.o
	g++ -o test_adaline main.cpp adaline.o adaline_network.o

adaline.o: adaline.cpp adaline.hpp
	g++ -c adaline.cpp

adaline_network.o: adaline_network.cpp adaline_network.hpp
	g++ -c adaline_network.cpp

interpolator : interpolador.cpp adaline.o adaline_network.o
	g++ -o test_interpolator interpolador.cpp adaline.o adaline_network.o

clean:
	rm *.o test_adaline test_interpolator

