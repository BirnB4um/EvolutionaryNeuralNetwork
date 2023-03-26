#include "EvolutionaryNeuralNetwork.h"

EvolutionaryNeuralNetwork::EvolutionaryNeuralNetwork() {
	number_of_input_nodes = 0;
	number_of_output_nodes = 0;
	number_of_hidden_nodes = 0;
	number_of_nodes = 0;
	number_of_connections = 0;
	number_of_activation_functions = 5;
	
	set_mutation_rates(10, 10, 10, 10, 10, 10, 10);

	min_weight = -2;
	max_weight = 2;
	min_bias = -4;
	max_bias = 4;
}

EvolutionaryNeuralNetwork::~EvolutionaryNeuralNetwork() {

}

void EvolutionaryNeuralNetwork::print() {
	std::cout << "=== Nodes ===\n";
	std::cout << "input: " << number_of_input_nodes << ", hidden: " << number_of_hidden_nodes << ", output: " << number_of_output_nodes << "\nnumber of connections: " << number_of_connections << "\n";
	std::cout << "\n--- Input Nodes: ---\n";
	for (int i = 0; i < number_of_input_nodes; i++) {
		std::cout << i << ":  " << nodes[i].bias << " bias   -   " << nodes[i].activation_function << " activation    (value: " << nodes[i].value << ")\n";
	}	
	std::cout << "--- Output Nodes: ---\n";
	for (int i = number_of_input_nodes; i < number_of_input_nodes + number_of_output_nodes; i++) {
		std::cout << i << ":  " << nodes[i].bias << " bias   -   " << nodes[i].activation_function << " activation    (value: " << nodes[i].value << ")\n";
	}
	std::cout << "--- Hidden Nodes: ---\n";
	for (int i = number_of_input_nodes + number_of_output_nodes; i < number_of_nodes; i++) {
		std::cout << i << ":  " << nodes[i].bias << " bias   -   " << nodes[i].activation_function << " activation    (value: " << nodes[i].value << ")\n";
	}	

	std::cout << "\n=== Connections ===\n\n";
	for (ConnectionPack &con_pack : connection_packs) {
		std::cout << "Node " << con_pack.node << ":\n";
		for (int n = 0; n < con_pack.input_nodes.size(); n++) {
			std::cout << "InputNode: " << con_pack.input_nodes[n] << " - weight: " << con_pack.input_weights[n] << "\n";
		}
		std::cout << "\n";
	}
	std::cout << "===================\n" << std::endl;
}


void EvolutionaryNeuralNetwork::set_mutation_rates(int change_weight_probability, int change_bias_probability, int change_activation_probability,
	int add_connection_probability, int remove_connection_probability, int add_node_probability, int remove_node_probability) {
	propablility_sum = 0;
	propablility_sum += change_weight_probability;
	this->change_weight_probability = propablility_sum;

	propablility_sum += change_bias_probability;
	this->change_bias_probability = propablility_sum;

	propablility_sum += change_activation_probability;
	this->change_activation_probability = propablility_sum;

	propablility_sum += add_connection_probability;
	this->add_connection_probability = propablility_sum;

	propablility_sum += remove_connection_probability;
	this->remove_connection_probability = propablility_sum;

	propablility_sum += add_node_probability;
	this->add_node_probability = propablility_sum;

	propablility_sum += remove_node_probability;
	this->remove_node_probability = propablility_sum;
}


bool EvolutionaryNeuralNetwork::create_from_code(std::vector<float>& code) {

	nodes.clear();
	connection_packs.clear();

	number_of_input_nodes = code[0];
	number_of_output_nodes = code[1];
	number_of_nodes = code[2];
	number_of_connections = code[3];
	number_of_hidden_nodes = number_of_nodes - number_of_input_nodes - number_of_output_nodes;

	if (number_of_input_nodes == 0 || number_of_output_nodes == 0) {
		return false;
	}

	//create nodes
	for (int i = 0; i < number_of_input_nodes; i++) {
		Node node;
		node.value = 0;
		node.activation_function = 0;
		node.bias = 0;
		nodes.push_back(node);
	}
	for (int i = 0; i < number_of_hidden_nodes + number_of_output_nodes; i++) {
		Node node;
		node.value = 0;
		node.bias = code[4 + i * 2];
		node.activation_function = code[4 + i * 2 + 1];
		nodes.push_back(node);
	}

	//create connections
	std::vector<Connection> all_connections;
	for (int i = 0; i < number_of_connections; i++) {
		Connection con;
		con.from = int(code[4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + i*3]);
		con.to = int(code[4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + i*3 + 1]);
		con.weight = code[4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + i*3 + 2];
		all_connections.push_back(con);
	}

	//pack connections together
	std::vector<ConnectionPack> unsorted_connections_packs;
	for (int i = number_of_input_nodes; i < number_of_nodes; i++) {
		ConnectionPack con_pack;
		con_pack.node = i;
		bool connection_exists = false;
		for (Connection &con : all_connections) {
			if (con.to == i) {
				connection_exists = true;
				con_pack.input_nodes.push_back(con.from);
				con_pack.input_weights.push_back(con.weight);
			}
		}
		if (connection_exists) {
			unsorted_connections_packs.push_back(con_pack);
		}
	}

	//sort connection_packs
	bool* marked_nodes = new bool[number_of_nodes];
	for (int i = 0; i < number_of_nodes;i++) {
		marked_nodes[i] = i < number_of_input_nodes;
	}

	bool node_marked = false;
	bool all_nodes_marked = false;
	while (!all_nodes_marked) {
		all_nodes_marked = true;
		node_marked = false;
		for (ConnectionPack &con_pack : unsorted_connections_packs) {
			if (!marked_nodes[con_pack.node]) {
				all_nodes_marked = false;
			
				bool all_inputs_marked = true;
				for (int &input_node : con_pack.input_nodes) {
					if (!marked_nodes[input_node]) {
						all_inputs_marked = false;
						break;
					}
				}
				if (all_inputs_marked) {
					node_marked = true;
					marked_nodes[con_pack.node] = true;
					connection_packs.push_back(con_pack);
				}

			}

		}

		//forcefully mark node
		if (!all_nodes_marked && !node_marked) {
			for (ConnectionPack& con_pack : unsorted_connections_packs) {
				if (!marked_nodes[con_pack.node]) {
					marked_nodes[con_pack.node] = true;
					connection_packs.push_back(con_pack);
					break;
				}
			}
		}

	}

	delete[] marked_nodes;
	optimise_building_code();

	return true;
}


void EvolutionaryNeuralNetwork::init(int input_nodes, int output_nodes) {
	number_of_input_nodes = input_nodes;
	number_of_output_nodes = output_nodes;
	number_of_hidden_nodes = 0;
	number_of_nodes = input_nodes + output_nodes;
	number_of_connections = 0;

	nodes.clear();
	connection_packs.clear();

	for (int i = 0; i < number_of_nodes; i++) {
		Node node;
		node.bias = 0;
		node.activation_function = TANH;
		node.value = 0;
		nodes.push_back(node);
	}

	optimise_building_code();
}


void EvolutionaryNeuralNetwork::optimise_building_code() {
	building_code.clear();

	building_code.push_back(number_of_input_nodes);
	building_code.push_back(number_of_output_nodes);
	building_code.push_back(number_of_nodes);
	int number_of_connections = 0;
	for (ConnectionPack &con_pack : connection_packs) {
		number_of_connections += con_pack.input_nodes.size();
	}
	building_code.push_back(number_of_connections);

	for (int i = number_of_input_nodes; i < number_of_nodes; i++) {
		building_code.push_back(nodes[i].bias);
		building_code.push_back(nodes[i].activation_function);
	}

	for (ConnectionPack &con_pack : connection_packs) {
		for (int i = 0; i < con_pack.input_nodes.size(); i++) {
			building_code.push_back(con_pack.input_nodes[i]);
			building_code.push_back(con_pack.node);
			building_code.push_back(con_pack.input_weights[i]);
		}
	}
}


void EvolutionaryNeuralNetwork::optimise_network() {

	bool* nodes_marked_from_input = new bool[number_of_nodes];
	bool* nodes_marked_from_output = new bool[number_of_nodes];

	std::vector<int> nodes_to_check;

	for (int i = 0; i < number_of_nodes; i++) {
		nodes_marked_from_input[i] = false;
		nodes_marked_from_output[i] = false;
	}

	//traverse from input
	for (int i = 0; i < number_of_input_nodes; i++) {
		nodes_to_check.push_back(i);
		nodes_marked_from_input[i] = true;
	}
	while (nodes_to_check.size() > 0) {
		for (ConnectionPack &con_pack : connection_packs) {
			if (!nodes_marked_from_input[con_pack.node]) {
				for (int &in : con_pack.input_nodes) {
					if (in == nodes_to_check[0]) {
						nodes_marked_from_input[con_pack.node] = true;
						nodes_to_check.push_back(con_pack.node);
						break;
					}
				}
			}
		}
		nodes_to_check.erase(nodes_to_check.begin());
	}

	//traverse from output
	for (int i = number_of_input_nodes; i < number_of_input_nodes + number_of_output_nodes; i++) {
		nodes_to_check.push_back(i);
		nodes_marked_from_output[i] = true;
	}
	while (nodes_to_check.size() > 0) {
		for (ConnectionPack& con_pack : connection_packs) {
			if (con_pack.node == nodes_to_check[0]) {
				for (int &in : con_pack.input_nodes) {
					if (!nodes_marked_from_output[in]) {
						nodes_marked_from_output[in] = true;
						nodes_to_check.push_back(in);
					}
				}
				break;
			}
		}
		nodes_to_check.erase(nodes_to_check.begin());
	}

	//mark all nodes that are connected from input to output
	bool* useful_nodes = nodes_marked_from_input;
	for (int i = number_of_input_nodes + number_of_output_nodes; i < number_of_nodes; i++) {
		useful_nodes[i] = nodes_marked_from_input[i] && nodes_marked_from_output[i] ? true : false;
	}
	for (int i = 0; i < number_of_input_nodes + number_of_output_nodes; i++) {
		useful_nodes[i] = true;
	}

	//remove unused nodes & connections
	for (int i = number_of_nodes - 1; i >= number_of_input_nodes + number_of_output_nodes; i--) {

		if (!useful_nodes[i]) {

			//remove node
			nodes.erase(nodes.begin() + i);
			number_of_hidden_nodes--;

			//remove connections to and from node
			for (int n = connection_packs.size() - 1; n >= 0; n--) {

				//remove con_pack
				if (connection_packs[n].node == i) {
					number_of_connections -= connection_packs[n].input_nodes.size();
					connection_packs.erase(connection_packs.begin() + n);
					continue;
				}

				//correct for remove-shift
				if (connection_packs[n].node > i) {
					connection_packs[n].node--;
				}

				for (int j = connection_packs[n].input_nodes.size() - 1; j >= 0; j--) {
					if (connection_packs[n].input_nodes[j] > i) {//correct for remove-shift
						connection_packs[n].input_nodes[j]--;
						continue;
					}

					if (connection_packs[n].input_nodes[j] == i) {
						number_of_connections--;
						connection_packs[n].input_nodes.erase(connection_packs[n].input_nodes.begin() + j);
						connection_packs[n].input_weights.erase(connection_packs[n].input_weights.begin() + j);
					}
				}
			}

		}

	}
	number_of_nodes = number_of_input_nodes + number_of_output_nodes + number_of_hidden_nodes;

	//TODO: add parallel connections together
	for (ConnectionPack &con_pack : connection_packs) {

	}


	delete[] nodes_marked_from_input;
	delete[] nodes_marked_from_output;
}


void EvolutionaryNeuralNetwork::set_input(unsigned int index, float value) {
	if (index < number_of_input_nodes) {
		nodes[index].value = value;
	}
}


void EvolutionaryNeuralNetwork::set_input(float *input) {
	for (int i = 0; i < number_of_input_nodes; i++) {
		nodes[i].value = input[i];
	}
}


float EvolutionaryNeuralNetwork::get_output(unsigned int index) {
	return nodes[number_of_input_nodes + index].value;
}


void EvolutionaryNeuralNetwork::clear_network() {
	for (Node& node : nodes) {
		node.value = 0;
	}
}


void EvolutionaryNeuralNetwork::forward() {

	for (ConnectionPack &con_pack : connection_packs) {
		float value = 0;
		for (int i = 0; i < con_pack.input_nodes.size(); i++) {
			value += con_pack.input_weights[i] * nodes[con_pack.input_nodes[i]].value;
		}
		value += nodes[con_pack.node].bias;

		
		if (nodes[con_pack.node].activation_function == TANH) {
			value = std::tanh(value);
		}
		else if (nodes[con_pack.node].activation_function == RELU) {
			value = value < 0 ? 0 : value;
		}
		else if (nodes[con_pack.node].activation_function == SIGMOID) {
			value = 1.0 / (1 + 1.0 / (std::exp(value)));
		}
		else if (nodes[con_pack.node].activation_function == STEP) {
			value = value < 0 ? 0 : 1;
		}

		nodes[con_pack.node].value = value;
	}

}


void EvolutionaryNeuralNetwork::mutate(int number_of_mutations) {
	if (number_of_nodes == 0) {
		return;
	}

	for (int mutation = 0; mutation < number_of_mutations; mutation++) {
		int r = rand() % propablility_sum;

		if (r < change_weight_probability) {
			change_weight();
		}
		else if (r < change_bias_probability) {
			change_bias();
		}
		else if (r < change_activation_probability) {
			change_activation();
		}
		else if (r < add_connection_probability) {
			add_connection();
		}
		else if (r < remove_connection_probability) {
			remove_connection();
		}
		else if (r < add_node_probability) {
			add_node();
		}
		else if (r < remove_node_probability) {
			remove_node();
		}
	}

	building_code[2] = number_of_nodes;
	building_code[3] = number_of_connections;
	create_from_code(building_code);
}


void EvolutionaryNeuralNetwork::mutate(int number_of_mutations, std::vector<float>& code) {

	building_code = std::vector<float>(code);

	number_of_input_nodes = building_code[0];
	number_of_output_nodes = building_code[1];
	number_of_nodes = building_code[2];
	number_of_connections = building_code[3];
	number_of_hidden_nodes = number_of_nodes - number_of_input_nodes - number_of_output_nodes;

	if (number_of_nodes == 0) {
		return;
	}

	for (int mutation = 0; mutation < number_of_mutations; mutation++) {
		int r = rand() % propablility_sum;

		if (r < change_weight_probability) {
			change_weight();
		}
		else if (r < change_bias_probability) {
			change_bias();
		}
		else if (r < change_activation_probability) {
			change_activation();
		}
		else if (r < add_connection_probability) {
			add_connection();
		}
		else if (r < remove_connection_probability) {
			remove_connection();
		}
		else if (r < add_node_probability) {
			add_node();
		}
		else if (r < remove_node_probability) {
			remove_node();
		}
	}

	building_code[2] = number_of_nodes;
	building_code[3] = number_of_connections;
	create_from_code(building_code);
}


void EvolutionaryNeuralNetwork::change_weight(){
	if (number_of_connections > 0) {
		int i = rand() % number_of_connections;
		building_code[4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + i * 3 + 2] = min_weight + (float(rand()) / RAND_MAX) * (max_weight - min_weight);
	}
}

void EvolutionaryNeuralNetwork::change_bias(){
	int i = rand() % (number_of_hidden_nodes + number_of_output_nodes);
	building_code[4 + i * 2] = min_bias + (float(rand()) / RAND_MAX) * (max_bias - min_bias);
}

void EvolutionaryNeuralNetwork::change_activation(){
	int i = rand() % (number_of_hidden_nodes + number_of_output_nodes);
	building_code[4 + i * 2 + 1] = rand() % number_of_activation_functions;
}

void EvolutionaryNeuralNetwork::remove_connection(){
	if (number_of_connections > 0) {
		int i = rand() % number_of_connections;
		building_code.erase(building_code.begin() + 4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + i * 3);
		building_code.erase(building_code.begin() + 4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + i * 3);
		building_code.erase(building_code.begin() + 4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + i * 3);
		number_of_connections--;
	}
}

void EvolutionaryNeuralNetwork::add_connection(){
	int in = rand() % (number_of_input_nodes + number_of_hidden_nodes);
	in = in < number_of_input_nodes ? in : in + number_of_output_nodes;
	int out = number_of_input_nodes + (rand() % (number_of_output_nodes + number_of_hidden_nodes));
	building_code.push_back(in);
	building_code.push_back(out);
	building_code.push_back(min_weight + (float(rand()) / RAND_MAX) * (max_weight - min_weight));
	number_of_connections++;
}

void EvolutionaryNeuralNetwork::remove_node(){
	if (number_of_hidden_nodes > 0) {
		int i = number_of_output_nodes + (rand() % number_of_hidden_nodes);
		building_code.erase(building_code.begin() + 4 + i * 2);
		building_code.erase(building_code.begin() + 4 + i * 2);
		number_of_nodes--;
		number_of_hidden_nodes--;
		int connections_to_remove = 0;
		i = i + number_of_input_nodes;
		for (int n = number_of_connections - 1; n >= 0; n--) {
			if (building_code[4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + n * 3] == i ||
				building_code[4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + n * 3 + 1] == i) {
				building_code.erase(building_code.begin() + 4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + n * 3);
				building_code.erase(building_code.begin() + 4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + n * 3);
				building_code.erase(building_code.begin() + 4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + n * 3);
				connections_to_remove++;
			}
			else {
				if (building_code[4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + n * 3] > i) {
					building_code[4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + n * 3] -= 1;
				}
				if (building_code[4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + n * 3 + 1] > i) {
					building_code[4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + n * 3 + 1] -= 1;
				}
			}
		}
		number_of_connections -= connections_to_remove;
	}
}

void EvolutionaryNeuralNetwork::add_node() {
	if (number_of_connections > 0) {
		int i = rand() % number_of_connections;
		building_code.push_back(number_of_nodes);
		building_code.push_back(building_code[4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + i * 3 + 1]);
		building_code.push_back(min_weight + (float(rand()) / RAND_MAX) * (max_weight - min_weight));
		building_code[4 + (number_of_hidden_nodes + number_of_output_nodes) * 2 + i * 3 + 1] = number_of_nodes;
		number_of_connections++;
		building_code.insert(building_code.begin() + 4 + (number_of_hidden_nodes + number_of_output_nodes) * 2, min_bias + (float(rand()) / RAND_MAX) * (max_bias - min_bias));
		building_code.insert(building_code.begin() + 4 + (number_of_hidden_nodes + number_of_output_nodes) * 2, rand() % number_of_activation_functions);
		number_of_nodes++;
		number_of_hidden_nodes++;
	}
}
