/*
	Simple Adaline Class Signatures
*/

#ifndef _ADALINE_HPP
#define _ADALINE_HPP

#include <vector>

class adaline{
	/*
		weights : 
			vector of long double representing the weights
			coming to that adaline
		etha : 
			long double representing the learning rate
	*/
	std::vector<long double> weights;
	long double etha;
	
public:
	/*
		Class constructor

		Receives:
			size:
				int representing the size of the input incoming to the adaline
			etha:
				long double representing the value of etha in the class
			weights_range:
				long double representing the range of the value of the incoming 
				weights [-weights_range, weights_range]
	*/
	adaline(int size, long double etha, long double weights_range);

	/*
		Operator *

		Receives:
			input:
				vector of long double representing the input of the adaline

		Returns:
			Long double representing the dot product of the weights and the input
	*/
	long double operator*(const std::vector<long double>& input);

	/*
		sigma function

		Receives:
			input:
				vector of long double representing the input of the adaline

		Returns:
			long double representing the sigma function evaluated at input
	*/	
	long double sigma(const std::vector<long double>& input);

	/*
		train function
			This function trains the adaline by changing it's weights

		Reveives:
			input:
				vector of long double representing the input data
			exp_output:
				value of the expected output of the given input
	*/
	void train(const std::vector<long double>& input, 
		       long double exp_output);
};

#endif