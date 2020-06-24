#include <iostream>
#include <vector>
#include <stdio.h>
#include "adaline.hpp"
#include "adaline_network.hpp"

// learning rate value
const long double ETHA1 = 0.00001;
const long double ETHA2 = 0.00003;
const long double ETHA3 = 0.00005;
const int num_pixels = 784;
const int output_size = 10;
const int num_cycles = 50;

int main(){
	FILE *input = fopen("mnist_train.csv", "r");

	std::vector< std::pair< std::vector<long double> , std::vector<long double> > > data;
	int ans;
	while(fscanf(input, "%d", &ans) != EOF){
		std::vector<long double> aux_ans(output_size,0.0);
		aux_ans[ans] = 1.0;

		std::vector<long double> pixels(num_pixels);

		for (int i = 0; i < num_pixels; ++i){
			int aux;
			fscanf(input, ",%d", &aux);
			pixels[i] = (long double)aux/255;
		}

		data.push_back({aux_ans, pixels});
	}
	fclose(input);

	adaline_network my_adaline1(output_size, ETHA1, 0.05, num_pixels);
	adaline_network my_adaline2(output_size, ETHA2, 0.05, num_pixels);
	adaline_network my_adaline3(output_size, ETHA3, 0.05, num_pixels);

	printf("Training with %Lf\n", ETHA1);
	my_adaline1.train_set(data, num_cycles);
	printf("Training with %Lf\n", ETHA2);
	my_adaline2.train_set(data, num_cycles);
	printf("Training with %Lf\n", ETHA3);
	my_adaline3.train_set(data, num_cycles);

	data.clear();

	input = fopen("mnist_test.csv", "r");
	while(fscanf(input, "%d", &ans) != EOF){
		std::vector<long double> aux_ans(output_size,0);
		aux_ans[ans] = 1;

		std::vector<long double> pixels(num_pixels);

		for (int i = 0; i < num_pixels; ++i){
			int aux;
			fscanf(input, ",%d", &aux);
			pixels[i] = (long double)aux/255;
		}

		data.push_back({aux_ans, pixels});
	}
	fclose(input);

	std::pair<int, int> results1 = my_adaline1.test_set(data);
	std::pair<int, int> results2 = my_adaline2.test_set(data);
	std::pair<int, int> results3 = my_adaline3.test_set(data);
	printf("With %Lf: %d correct of %d, %.2Lf%% \n", ETHA1, results1.first, results1.second, (long double)(100*results1.first)/results1.second);
	printf("With %Lf: %d correct of %d, %.2Lf%% \n", ETHA2, results2.first, results2.second, (long double)(100*results2.first)/results2.second);
	printf("With %Lf: %d correct of %d, %.2Lf%% \n", ETHA3, results3.first, results3.second, (long double)(100*results3.first)/results3.second);
	return 0;
}