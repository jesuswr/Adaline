/*
	Simple Adaline Class Implementation
*/

#include <vector>
#include <iostream>
#include <stdlib.h>
#include "adaline.hpp"

/*
	Class constructor
		Reserves the size for the weights of the adaline and
		asigns random values to them in the given range.
	 	The bias is in the last space of the vector.
*/
adaline::adaline(int size, long double etha, 
	                   long double weights_range) : etha(etha) {
	weights.resize(size+1);
	for (int i = 0; i <= size; ++i){
		long double rnd = (long double)rand()*2/RAND_MAX -1;
		weights[i] = (long double)rnd*weights_range;
	}
}

/*
	Operator *
		Basic dot product implementation
*/
long double adaline::operator*(const std::vector<long double>& input){
	long double acum = 0;
	int n = input.size();
	for (int i = 0; i < n; ++i){
		acum += weights[i]*input[i];
	}
	// Add the bias value
	acum += weights[n];
	return acum;
}

/*
	sigma function
		Calculates the doct product and returns it
*/
long double adaline::sigma(const std::vector<long double>& input){
	return (*this)*input ;
}

/*
	train function
		Trains the adaline by changing the weights depending on the
		cuadratic difference
*/
void adaline::train(const std::vector<long double>& inp, 
	                   long double exp_outp){
	long double outp = sigma(inp);
	long double err = exp_outp - outp;
	
	int n = inp.size();
	for (int i = 0; i < n; ++i){
		weights[i] += err*inp[i]*etha;
	}
	// Adjust bias value
	weights[n] += err*etha;
}
