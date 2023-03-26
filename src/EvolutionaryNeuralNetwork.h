#pragma once
#include <vector>
#include <iostream>
#include <math.h>


/*

=== Building Code: ===

	- 1 float: number of input nodes
	- 1 float: number of output nodes
	- 1 float: number of total nodes
	- 1 float: number of connections

	- n times (n = total nodes - input nodes): (2 floats per node) (first output nodes then hidden nodes)
		- 1 float: node biases
		- 1 float: node actiation functions

	- n times (n = number of connections): (3 floats per connection)
		- 1 floats: connection from
		- 1 floats: connection to
		- 1 floats: connection weight


=== Activation Functions: ===
	- 0: TANH
	- 1: IDENTITY
	- 2: RELU
	- 3: SIGMOID
	- 4: STEP


=== Mutation: ===
	-create connection between two existing nodes (random weight)
	-remove connection
	-change connection weight
	-create node on existing connection (second connection has random weight)
	-remove node (removes all connections to and from this node)
	-change node bias
	-change node activation function

*/




class EvolutionaryNeuralNetwork
{
public:

	struct Node {
		float value;
		float bias;
		int activation_function;
	};

	struct Connection {
		int from;
		int to;
		float weight;
	};

	struct ConnectionPack {
		int node = 0;
		std::vector<int> input_nodes;
		std::vector<float> input_weights;
	};

	const static enum ACTIVATION {
		TANH, IDENTITY, RELU, SIGMOID, STEP
	};


	std::vector<float> building_code;
	int number_of_input_nodes;
	int number_of_output_nodes;
	int number_of_hidden_nodes;
	int number_of_nodes;
	int number_of_connections;

	float min_weight, max_weight;
	float min_bias, max_bias;

	std::vector<Node> nodes;
	std::vector<ConnectionPack> connection_packs;
	

	EvolutionaryNeuralNetwork();
	~EvolutionaryNeuralNetwork();

	//build empty network
	void init(int input_nodes, int output_nodes);

	//build network via code
	bool create_from_code(std::vector<float> &code);

	//clear all node-values
	void clear_network();

	//create the standart code for the current network state
	void optimise_building_code();

	//remove parts of network that dont contribute anything to the output and add same connections together
	//NOTE: this doesnt change the building code. call optimise_building_code() to change it afterwards
	void optimise_network();

	//set probabilities for each mutation to occure
	void set_mutation_rates(int change_weight_probability, int change_bias_probability, int change_activation_probability, 
							int add_connection_probability, int remove_connection_probability, int add_node_probability, int remove_node_probability);
	//mutate network
	void mutate(int number_of_mutations);
	//mutate given code and build network from this code
	void mutate(int number_of_mutations, std::vector<float> &code);
	//forward input through network to output
	void forward();

	//index must be within range of number of input nods
	void set_input(unsigned int index, float value);
	//input size has to match number of input nodes
	void set_input(float *input);
	//index must be within range of number of output nodes
	float get_output(unsigned int index);
	//output size has to match number of output nodes
	//void get_output(float *output);

	//print whole network to console
	void print();


private:

	int number_of_activation_functions;

	int propablility_sum;
	int change_weight_probability;
	int change_bias_probability;
	int change_activation_probability;
	int add_connection_probability;
	int remove_connection_probability;
	int add_node_probability;
	int remove_node_probability;

	void change_weight();
	void change_bias();
	void change_activation();
	void remove_connection();
	void add_connection();
	void remove_node();
	void add_node();


};

