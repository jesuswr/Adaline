#include <iostream>
#include <vector>
#include <stdio.h>
#include "adaline.hpp"
#include "adaline_network.hpp"

const int POL_GRADE = 8;
const int NUM_CYCLES = 10000;
const long double ETHA = 0.000001;

int main(){
	FILE *input = fopen("datosT3.csv", "r");
	std::vector< std::pair< std::vector<long double> , std::vector<long double> > > data;
	long double a,b;
	while( fscanf(input, "%Lf,%Lf", &b, &a) != EOF ){
		std::vector<long double> aux_ans = { a };
		std::vector<long double> pol(POL_GRADE);

		pol[0] = b;
		for (int i = 1; i < POL_GRADE; ++i){
			pol[i] = pol[i-1] * b;
		}

		data.push_back({aux_ans, pol});
	}
	adaline_network my_adaline(1, ETHA, 0.05, POL_GRADE);
	printf("Training\n");
	my_adaline.train_set(data, NUM_CYCLES);

	long double err = my_adaline.test_set2(data);	
	printf("Total error of %Lf\n", err);
	return 0;
}