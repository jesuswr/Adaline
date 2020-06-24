/*
	Signatures of a Simple Adaline Network Class
*/

#ifndef _ADALINE_NETWORK_HPP
#define _ADALINE_NETWORK_HPP

#include <vector>
#include "adaline.hpp"

class adaline_network{
	/*
		adal:
			vector of adalines representing the layer of adalines
			in a single layer model
	*/
	std::vector<adaline> adal;

public:
	/*
		Class constructor

		Receives:
			n:
				int representing the number of adalines
			etha:
				long double representing the value of etha in the class
			weights_range:
				long double representing the range of the value of the  
				weights [-weights_range, weights_range]
			inp_size:
				int representing the size of the input incoming
	*/
	adaline_network(int n, long double etha, long double w_range, int inp_size);

	/*
		train function
			This function trains the adaline layer

		Receives:
			inp:
				vector of long double representing the input data
			ans:
				vector of long double representing the expected output of the layer
	*/
	void train(const std::vector<long double>& inp, const std::vector<long double>& ans);

	/*
		train_set function
			This function trains the adaline layer with a set of data

		Receives:
			data:
				vector of pairs, first element of the pair is a vector of long doubles
				that represents the expected output for the layer and the
				second element is a vector of long double representing the input
			cycles:
				int that represents the number of epochs for the training,
				if not set then = 1

	*/
	void train_set(const std::vector< std::pair<std::vector<long double>, 
		                                        std::vector<long double> > >& data, 
		                                        int cycles = 1);

	/*
		test function
			Receives input and expected output and returns true if the 
			output of the layer is the same as the expected output

		Receives:
			inp:
				vector of long double representing the input data
			ans:
				vector of long double representing the expected output

		Returns:
			true if the output equals the expected output or false otherwise
	*/
	bool test(const std::vector<long double>& inp, const std::vector<long double>& ans);

	/*
		test_set function
			This function tests the adaline layer with a set of data

		Receives:
			data:
				vector of pairs, first element of the pair is a vector of long doubles
				that represents the expected output for the layer and the
				second element is a vector of long double representing the input
		
		Returns:
			A pair of int, the first element is the number of testing elements
			that were classified correctly and the second element is the total
			number of testing elements
	*/
	std::pair<int,int> test_set(const std::vector< std::pair<std::vector<long double>, 
		                                                     std::vector<long double> > >& data);


	/*
		test_set2 function
			This function tests the adaline layer with a set of data

		Receives:
			data:
				vector of pairs, first element of the pair is a vector of long doubles
				that represents the expected output for the layer and the
				second element is a vector of long double representing the input
		
		Returns:
			long double that represents the cuadratic error
	*/
	long double test_set2(const std::vector< std::pair<std::vector<long double>, 
		                                                     std::vector<long double> > >& data);
};

#endif