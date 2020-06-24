/*
	Simple Implementation of a Adaline Network Class
*/

#include <time.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include "adaline.hpp"
#include "adaline_network.hpp"

/*
	Class constructor
		Reserves space for the adalines and calls their constructor
*/
adaline_network::adaline_network(int n, long double etha, 
	                                   long double w_r, int inp_sz){
	srand(time(NULL));
	for (int i = 0; i < n; ++i){
		adaline aux(inp_sz, etha, w_r);
		adal.push_back(aux);
	}
}

/*
	train function
		Trains each adaline with the given data 
*/
void adaline_network::train(const std::vector<long double>& inp, 
	                           const std::vector<long double>& ans){
	int n = adal.size();
	for (int i = 0; i < n; ++i)
		adal[i].train(inp, ans[i]);
	
}

/*
	train_set function
		Given a set of training data, trains each adaline with each test
*/
void adaline_network::train_set(const std::vector< std::pair<std::vector<long double>, 
	                                                         std::vector<long double> > >& data, 
	                                                            int cycles){
	int n = data.size();
	for (int k = 0; k < cycles; ++k){
		for (int i = 0; i < n; ++i){
			train(data[i].second, data[i].first);
		}
		printf("Training cycle #%d complete\n", k+1);
	}
}

/*
	test function
		Looks for the maximun value of the adalines, if
		the adaline that represents the number i is max 
		and the expected answer is i returns true else false
*/
bool adaline_network::test(const std::vector<long double>& inp, 
	                          const std::vector<long double>& ans){
	int n = ans.size();
	long double max_output = -10000.0;
	int max_ind;
	for (int i = 0; i < n; ++i){
		long double curr_output = adal[i].sigma(inp);
		if ( max_output < curr_output ){
			max_output = curr_output;
			max_ind = i;
		}
	}

	return ( ans[max_ind] > 0.99 );
}

/*
	test_set function
		Counts how much data is classified correctly
*/
std::pair<int,int> adaline_network::test_set(const std::vector< std::pair<std::vector<long double>, 
	                                                                         std::vector<long double> > >& data){
	int n = data.size();
	int correct = 0;
	for (int i = 0; i < n; ++i){
		if ( (*this).test(data[i].second, data[i].first) )
			correct++;
	}
	return {correct, n};
}

/*
	test_set2 function
		Calculates the cuadratic error for the interpolation problem
*/
long double adaline_network::test_set2(const std::vector< std::pair<std::vector<long double>, 
	                                                                         std::vector<long double> > >& data){
	int n = data.size();
	long double err = 0;
	for (int i = 0; i < n; ++i){
		long double out = adal[0].sigma(data[i].second);
		long double exp_out = data[i].first[0];
		err += (out - exp_out)*(out - exp_out);
		printf("With %Lf should answer %Lf and it answers %Lf, error of %Lf\n",
			data[i].second[0], exp_out, out , out - exp_out);
	}
	return err/2;
}