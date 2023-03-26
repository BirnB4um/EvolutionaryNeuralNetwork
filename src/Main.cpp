#include <iostream>
#include "EvolutionaryNeuralNetwork.h"


struct Network {
	EvolutionaryNeuralNetwork nn;
	float error;
};


int main() {

	//Test the Network in the XOR example


	std::srand(896435);

	int epoche = 0;
	int max_number_of_mutations = 5;

	std::vector<Network> networks_list;
	std::vector<int> sorted_networks_index_list;

	int number_of_networks = 500;
	for (int i = 0; i < number_of_networks; i++) {
		Network network;
		network.nn.init(2, 1);

		network.error = 4;

		networks_list.push_back(network);
		sorted_networks_index_list.push_back(i);
	}

	//training data
	float training_data[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
	float training_answer[4] = {    0,     1,     1,     0 };

	while (true) {
		epoche++;
		std::cout << "Epoche: " << epoche << "  |  ";

		//sort networks by error
		for (int max_i = number_of_networks - 1; max_i >= number_of_networks / 2; max_i--) {
			for (int i = 0; i < max_i; i++) {
				if (networks_list[sorted_networks_index_list[i]].error < networks_list[sorted_networks_index_list[i+1]].error) {
					int temp = sorted_networks_index_list[i];
					sorted_networks_index_list[i] = sorted_networks_index_list[i + 1];
					sorted_networks_index_list[i + 1] = temp;
				}
			}
		}

		//print best network to console
		std::cout << "error: " << networks_list[sorted_networks_index_list[number_of_networks - 1]].error << std::endl;

		if (networks_list[sorted_networks_index_list[number_of_networks - 1]].error == 0) {
			std::cout << "\n";
			networks_list[sorted_networks_index_list[number_of_networks - 1]].nn.print();

			for (int test_i = 0; test_i < 4; test_i++) {
				networks_list[sorted_networks_index_list[number_of_networks - 1]].nn.clear_network();
				networks_list[sorted_networks_index_list[number_of_networks - 1]].nn.set_input(0, training_data[test_i][0]);
				networks_list[sorted_networks_index_list[number_of_networks - 1]].nn.set_input(1, training_data[test_i][1]);
				networks_list[sorted_networks_index_list[number_of_networks - 1]].nn.forward();

				std::cout << training_data[test_i][0] << ", " << training_data[test_i][1] << "  ->  ";
				std::cout << networks_list[sorted_networks_index_list[number_of_networks - 1]].nn.get_output(0) << std::endl;;

			}

			break;
		}


		//mutate lower half based on upper half
		for (int i = 0; i < number_of_networks / 2; i++) {
			networks_list[sorted_networks_index_list[i]].nn.mutate(1 + (rand() % max_number_of_mutations),
						networks_list[sorted_networks_index_list[number_of_networks - 1 - i]].nn.building_code);
		}


		//reset score for next epoche
		for (Network &network : networks_list) {
			network.error = 0;
		}
		
		//test networks
		for (int input_i = 0; input_i < 4; input_i++) {

			for (Network &network : networks_list) {
				network.nn.clear_network();

				network.nn.set_input(0, training_data[input_i][0]);
				network.nn.set_input(1, training_data[input_i][1]);

				network.nn.forward();

				network.error += std::abs( training_answer[input_i] - network.nn.get_output(0) );

			}

		}

	}



	return 0;
}