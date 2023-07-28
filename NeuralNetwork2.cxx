#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <cmath>
#include <thread>
#include <atomic>
#include <mutex>
/*
	My data structure for neural network
	[
	[number of layers in the neural network],[
		[number of neurons in layer1],[ [length of neuron1],[activation],[weight1],[weight2]....[bias]],			[[length of neuron2],[activation],[weight1],[weight2]....[bias] ]
.....],
	[number of layers in the neural network],[
		[number of neurons in layer2],[ [length of neuron1],[activation],[weight1],[weight2]....[bias]],			[[length of neuron2],[activation],[weight1],[weight2]....[bias] ]
.....],
.
.
.
	]
*/
void nn_clear(double ***nn, double default_value = 0)
{
	for (int layer_no = 1; layer_no < ***nn; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < ***(nn + layer_no); neuron_no++)
		{
			for (int weight_no = 2; weight_no < *(*(*(nn + layer_no) + neuron_no)); weight_no++)
			{
				*(*(*(nn + layer_no) + neuron_no) + weight_no) = default_value;
			}
		}
	}
}
double ***nn_create(const int *neuron_cts)
{
	int prev_neuron_no = -1;
	double ***nn = (double ***)(malloc((neuron_cts[0]) * 8));
	if (nn == NULL)
	{
		printf("\nERROR: Memory was not allocated by OS to create the neural network\nnn_create() Aborted\n");
		exit(1);
		return nn;
	}
	nn[0] = (double **)(malloc(8));
	nn[0][0] = (double *)(malloc(8));
	nn[0][0][0] = neuron_cts[0];
	for (int layer_no = 1; layer_no < neuron_cts[0]; layer_no++)
	{
		nn[layer_no] = (double **)(malloc((neuron_cts[layer_no] + 1) * 8));
		if (nn[layer_no] == NULL)
		{
			printf("\nERROR: Memory was not allocated by OS to create the neural network\nnn_create() Aborted\n");
			return nn;
			exit(1);
		}
		nn[layer_no][0] = (double *)(malloc(8));
		nn[layer_no][0][0] = neuron_cts[layer_no] + 1;
		for (int neuron_no = 1; neuron_no <= neuron_cts[layer_no]; neuron_no++)
		{
			nn[layer_no][neuron_no] = (double *)(malloc((prev_neuron_no + 3) * 8));
			if (nn[layer_no][neuron_no] == NULL)
			{
				printf("\nERROR: Memory was not allocated by OS to create the neural network\nnn_create() Aborted\n");
				exit(1);
				return nn;
			}
			nn[layer_no][neuron_no][0] = prev_neuron_no + 2 + 1;
		}
		prev_neuron_no = neuron_cts[layer_no];
	}
	return nn;
}
inline int nn_layerCount(double ***nn)
{
	return int(nn[0][0][0]) - 1;
}
inline int nn_neuronCount(double ***nn, int layer_no)
{
	if ((layer_no > 0) and (layer_no < nn[0][0][0]))
		return int(nn[layer_no][0][0]) - 1;
	else
	{
		printf("\nERROR: The neural network had %i layers but number of neurons of %ith layer was asked\nnn_neuronCount() Aborted\n", nn_layerCount(nn), layer_no);
		exit(1);
		return -1;
	}
}
void nn_show(double ***nn)
{
	printf("Number of layers in this neural network = %i\n", (int)(nn[0][0][0] - 1));
	printf("\n<<Layer 1 (input layer) has %i neurons>>\n\n", (int)(nn[1][0][0] - 1));
	for (int neuron_no = 1; neuron_no < nn[1][0][0]; neuron_no++)
	{
		printf("Neuron %i Activation- %f\n", neuron_no, nn[1][neuron_no][1]);
	}
	for (int layer_no = 2; layer_no < nn[0][0][0]; layer_no++)
	{
		if (layer_no != nn[0][0][0] - 1)
			printf("\n<<Layer %i has %i neurons>>\n\n", layer_no, (int)(nn[layer_no][0][0] - 1));
		else
			printf("\n<<Layer %i (output layer) has %i neurons>>\n\n", layer_no, (int)(nn[layer_no][0][0] - 1));
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			printf("Neuron %i\nActivation- %f\nWeights- [", neuron_no, nn[layer_no][neuron_no][1]);
			int weight_no = 2;
			for (; weight_no < nn[layer_no][neuron_no][0] - 1; weight_no++)
			{
				printf(" %f,", nn[layer_no][neuron_no][weight_no]);
			}
			printf("]\nBias- %f\n\n", nn[layer_no][neuron_no][weight_no]);
		}
	}
	printf("Network finished!\n");
}
inline double ***nn_dublicate(double ***nn)
{
	int neuron_cts[int(nn[0][0][0])];
	neuron_cts[0] = int(nn[0][0][0]);
	for (int layer_no = 1; layer_no < neuron_cts[0]; layer_no++)
	{
		neuron_cts[layer_no] = nn_neuronCount(nn, layer_no);
	}
	return nn_create(neuron_cts);
}
void nn_copy(double ***nn2, double ***nn)
{
	for (int layer_no = 1; (layer_no < ***nn) && (layer_no < ***nn2); layer_no++)
	{
		for (int neuron_no = 1; (neuron_no < ***(nn + layer_no)) && (neuron_no < ***(nn2 + layer_no)); neuron_no++)
		{
			for (int weight_no = 1; (weight_no < **(*(nn + layer_no) + neuron_no)) && (weight_no < **(*(nn2 + layer_no) + neuron_no)); weight_no++)
			{
				*(*(*(nn2 + layer_no) + neuron_no) + weight_no) = *(*(*(nn + layer_no) + neuron_no) + weight_no);
			}
		}
	}
}
void nn_delete(double ***nn)
{
	for (int layer_no = 1; layer_no < nn[0][0][0]; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			free(nn[layer_no][neuron_no]); // = (double *)(malloc((prev_neuron_no + 3) * 8));
		}
		free(nn[layer_no][0]); // = (double *)(malloc(8));
		free(nn[layer_no]);	   // = (double **)(malloc((neuron_cts[layer_no] + 1) * 8));
	}
	free(nn[0][0]); // = (double *)(malloc(8));
	free(nn[0]);	// = (double **)(malloc(8));
	free(nn);		// = (double ***)(malloc((neuron_cts[0] + 1) * 8));
	nn = NULL;
}
inline void nn_resize(double ***nn, int *neuron_cts)
{
	double ***temp_nn = nn_create(neuron_cts);
	nn_copy(temp_nn, nn);
	nn_delete(nn);
	nn = nn_create(neuron_cts);
	nn_copy(nn, temp_nn);
}
double nn_LEAK = 0.01, nn_LEARNING_RATE = 0.01;
double nn_SLOPE_CHANGING_THRESHOLD = -nn_LEAK * nn_LEARNING_RATE;
inline void nn_set_learningRate(double learning_rate)
{
	if (learning_rate <= 0)
	{
		printf("\nERROR: Non-positive value was passed to nn_set_learningRate\nnn_set_learningRate() Aborted\n");
		exit(1);
	}
	nn_LEARNING_RATE = learning_rate;
	nn_SLOPE_CHANGING_THRESHOLD = -nn_LEAK * nn_LEARNING_RATE;
}
inline void nn_set_leak(double leak)
{
	nn_LEAK = leak;
	nn_SLOPE_CHANGING_THRESHOLD = -nn_LEAK * nn_LEARNING_RATE;
}
inline double nn_activationf(double activation)
{
	if (activation < 0)
		return activation * nn_LEAK;
	return activation;
}
inline double nn_aActivationDerivativef(double activation)
{
	if (activation >= 0)
		return 1;
	if (activation < nn_SLOPE_CHANGING_THRESHOLD)
		return nn_LEAK;
	return (((activation / nn_LEAK) + nn_LEARNING_RATE) - activation) / nn_LEARNING_RATE;
}
inline double nn_outputActivationf(double activation)
{
	return 1 / (pow(2.7, -activation) + 1); //sigmoid
}
inline double nn_aOutputActivationDerivativef(double activation)
{
	return activation - activation * activation; //for sigmoid
}
inline void nn_cmpt_activation_hidden(double **prev_layer, double *neuron)
{
	double activation = 0;
	int neuron_no = 1;
	for (; neuron_no < **prev_layer; neuron_no++)
	{
		activation += *(*(prev_layer + neuron_no) + 1) * (*(neuron + neuron_no + 1)); //Adding weigted sum
	}
	activation += *(neuron + neuron_no + 1); //Adding bias
	*(neuron + 1) = nn_activationf(activation);
}
inline void nn_cmpt_activation_output(double **prev_layer, double *neuron)
{
	double activation = 0;
	int neuron_no = 1;
	for (; neuron_no < **prev_layer; neuron_no++)
	{
		activation += *(*(prev_layer + neuron_no) + 1) * (*(neuron + neuron_no + 1)); //Adding weigted sum
	}
	activation += *(neuron + neuron_no + 1); //Adding bias
	*(neuron + 1) = nn_outputActivationf(activation);
}
double **nn_process(double ***nn)
{
	int layer_no = 2;
	for (; (layer_no + 1) < ***nn; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < ***(nn + layer_no); neuron_no++)
		{
			nn_cmpt_activation_hidden(*(nn + layer_no - 1), *(*(nn + layer_no) + neuron_no));
		}
	}
	for (int neuron_no = 1; neuron_no < ***(nn + layer_no); neuron_no++)
	{
		nn_cmpt_activation_output(*(nn + layer_no - 1), *(*(nn + layer_no) + neuron_no));
	}
	return *(nn + (int)(***nn) - 1);
}
void nn_input(double ***nn, double *input)
{
	for (int neuron_no = 1; (neuron_no < *input) && (neuron_no < ***(nn + 1)); neuron_no++)
	{
		*(*(*(nn + 1) + neuron_no) + 1) = *(input + neuron_no);
	}
}
void nn_discreatOutput(double **layer, bool *discreatOutput, double threshold = 0.5)
{
	for (int neuron_no = 1; neuron_no < layer[0][0]; neuron_no++)
	{
		if (layer[neuron_no][1] > threshold)
			discreatOutput[neuron_no] = 1;
		else
			discreatOutput[neuron_no] = 0;
	}
}
void nn_save(double ***nn, const char *path)
{
	std::ofstream saving_file(path);
	saving_file << nn[0][0][0] << '\n';
	for (int layer_no = 1; layer_no < nn[0][0][0]; layer_no++)
	{
		saving_file << nn[layer_no][0][0] << '\n';
	}
	for (int layer_no = 1; layer_no < nn[0][0][0]; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			for (int weight_no = 1; weight_no < nn[layer_no][neuron_no][0]; weight_no++)
			{
				saving_file << nn[layer_no][neuron_no][weight_no] << ' ';
			}
		}
	}
}
double ***nn_load(const char *path)
{
	std::string word;
	std::ifstream loading_file(path);
	loading_file >> word;
	int neuron_cts[stoi(word)];
	neuron_cts[0] = stoi(word);
	for (int layer_no = 1; layer_no < neuron_cts[0]; layer_no++)
	{
		loading_file >> word;
		neuron_cts[layer_no] = stoi(word) - 1;
	}
	double ***nn = nn_create(neuron_cts);
	for (int layer_no = 1; layer_no < nn[0][0][0]; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			for (int weight_no = 1; weight_no < nn[layer_no][neuron_no][0]; weight_no++)
			{
				loading_file >> word;
				nn[layer_no][neuron_no][weight_no] = stod(word);
			}
		}
	}
	return nn;
}
double nn_MAX_WEIGHT = 100;
void nn_set_maxWeight(double maxWeight)
{
	if (maxWeight < 0)
	{
		printf("\nERROR: Negative value was passed to nn_set_maxWeight\nnn_set_maxWeight() Aborted\n");
		exit(1);
	}
	nn_MAX_WEIGHT = maxWeight;
}
void nn_mutate(double ***nn, double probability, double max_dweight)
{
	for (int layer_no = 2; layer_no < ***nn; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < ***(nn + layer_no); neuron_no++)
		{
			for (int weight_no = 2; weight_no < *(*(*(nn + layer_no) + neuron_no)); weight_no++)
			{
				double random_dweight = max_dweight * ((2 * (double)(rand()) / INT_MAX) - 1);
				if (((double)(rand()) / INT_MAX) < probability)
					*(*(*(nn + layer_no) + neuron_no) + weight_no) += random_dweight;
				if (abs(*(*(*(nn + layer_no) + neuron_no) + weight_no)) > nn_MAX_WEIGHT)
					*(*(*(nn + layer_no) + neuron_no) + weight_no) -= random_dweight;
			}
		}
	}
}
void nn_mutate_layer(double ***nn, double probability, double max_dweight, int layer_no)
{
	if (layer_no > nn_layerCount(nn))
	{
		printf("\nWARNING: Cannot mutate %ith layer. NOTE the passed nn has %i layers only\nnn_mutate_layer() Failed\n", layer_no, nn_layerCount(nn));
		return;
	}
	if (layer_no <= 1)
	{
		printf("\nWARNING: Cannot mutate %ith layer. NOTE 1st layer ( input layer ) doesn't contain weights but output layer do\nnn_mutate_layer() Failed\n", layer_no);
		return;
	}
	for (int neuron_no = 1; neuron_no < ***(nn + layer_no); neuron_no++)
	{
		for (int weight_no = 2; weight_no < *(*(*(nn + layer_no) + neuron_no)); weight_no++)
		{
			double random_dweight = max_dweight * ((2 * (double)(rand()) / INT_MAX) - 1);
			if (((double)(rand()) / INT_MAX) < probability)
				*(*(*(nn + layer_no) + neuron_no) + weight_no) += random_dweight;
			if (abs(*(*(*(nn + layer_no) + neuron_no) + weight_no)) > nn_MAX_WEIGHT)
				*(*(*(nn + layer_no) + neuron_no) + weight_no) -= random_dweight;
		}
	}
}
void nn_applyGamma(double ***gradient, double gamma)
{
	for (int layer_no = 2; layer_no < ***gradient; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < ***(gradient + layer_no); neuron_no++)
		{
			for (int weight_no = 2; weight_no < *(*(*(gradient + layer_no) + neuron_no)); weight_no++)
			{
				gradient[layer_no][neuron_no][weight_no] *= gamma;
			}
		}
	}
}
double nn_cost(double ***nn, const double *expectedOutput)
{
	double cost = 0;
	double **outputLayer = nn[int(nn[0][0][0]) - 1];
	for (int neuron_no = 1; (neuron_no < outputLayer[0][0]) and (neuron_no < expectedOutput[0]); neuron_no++)
	{
		cost += (outputLayer[neuron_no][1] - expectedOutput[neuron_no]) * (outputLayer[neuron_no][1] - expectedOutput[neuron_no]);
	}
	return cost;
}
void nn_derivate(double ***nn, double *expectedOutput, double ***gradient) //Replace activation of each neuron of a neural net by d(Cost)/d(Activation of that neuron)
{
	//derivating output layer
	for (int neuron_no = 1; neuron_no < nn[int(nn[0][0][0]) - 1][0][0]; neuron_no++)
	{
		gradient[int(nn[0][0][0]) - 1][neuron_no][1] = 2 * (nn[int(nn[0][0][0]) - 1][neuron_no][1] - expectedOutput[neuron_no]);
	}
	//derivating rest of the layers
	int layer_no = int(nn[0][0][0]) - 2;

	for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
	{
		gradient[layer_no][neuron_no][1] = 0;
		for (int nextLayer_neuron_no = 1; nextLayer_neuron_no < nn[layer_no + 1][0][0]; nextLayer_neuron_no++)
		{
			gradient[layer_no][neuron_no][1] += nn[layer_no + 1][nextLayer_neuron_no][neuron_no + 1] * gradient[layer_no + 1][nextLayer_neuron_no][1] * nn_aOutputActivationDerivativef(nn[layer_no + 1][nextLayer_neuron_no][1]);
		}
	}

	for (layer_no = int(nn[0][0][0]) - 3; layer_no > 1; layer_no--)
	{
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			gradient[layer_no][neuron_no][1] = 0;
			for (int nextLayer_neuron_no = 1; nextLayer_neuron_no < nn[layer_no + 1][0][0]; nextLayer_neuron_no++)
			{
				gradient[layer_no][neuron_no][1] += nn[layer_no + 1][nextLayer_neuron_no][neuron_no + 1] * gradient[layer_no + 1][nextLayer_neuron_no][1] * nn_aActivationDerivativef(nn[layer_no + 1][nextLayer_neuron_no][1]); /*Using the formula activation of d(Cost)/d(Activation of neuron of layer no = L) = d(Cost)/d(Activation of neuron of layer no L-1)*d(Activation of neuron of layer no L-1)/d(Activation of neuron of layer no L)
= sigma{over all the neurons of next layer}(weight connecting a neuron of next layer*d(Cost)/d(Activation of that neuron of layer no L) * derivative of activation function)*/
			}
		}
	}
	//derivating weights
	layer_no = 2;
	for (; layer_no < nn[0][0][0] - 1; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			int weight_no = 2;
			for (; weight_no < nn[layer_no][neuron_no][0] - 1; weight_no++)
			{
				gradient[layer_no][neuron_no][weight_no] += nn[layer_no - 1][weight_no - 1][1] * gradient[layer_no][neuron_no][1] * nn_aActivationDerivativef(nn[layer_no][neuron_no][1]);
			}
			gradient[layer_no][neuron_no][weight_no] += gradient[layer_no][neuron_no][1] * nn_aActivationDerivativef(nn[layer_no][neuron_no][1]);
		}
	}

	for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
	{
		int weight_no = 2;
		for (; weight_no < nn[layer_no][neuron_no][0] - 1; weight_no++)
		{
			gradient[layer_no][neuron_no][weight_no] += nn[layer_no - 1][weight_no - 1][1] * gradient[layer_no][neuron_no][1] * nn_aOutputActivationDerivativef(nn[layer_no][neuron_no][1]);
		}
		gradient[layer_no][neuron_no][weight_no] += gradient[layer_no][neuron_no][1] * nn_aOutputActivationDerivativef(nn[layer_no][neuron_no][1]);
	}
}
class nn_thread;
class nn_trainingData
{
	friend void nn_decendGradient(nn_thread *, nn_trainingData&, double);
	friend void nn_decendGradient(double ***, nn_trainingData&);

  private:
	double ***gradient;
	int inserted = 0;
	int size;

  public:
	double **inputs;
	double **outputs;
	nn_trainingData(double ***nn, int arg_size)
	{
		size = arg_size;
		if (size > 0)
		{
			inputs = (double **)malloc(8 * size);
			outputs = (double **)malloc(8 * size);//printf("inputs=%p,outputs=%p\n",inputs,outputs);
			gradient = nn_dublicate(nn);
		}
		else
		{
			printf("\nERROR: Size passed in nn_trainingData should be a positive integer. No memory was Allocated for nn_trainingData\nnn_trainingData::constructor() Aborted\n");
			exit(1);
		}
	}
	inline int get_size()
	{
		return size;
	}
	inline int get_insertionCount()
	{
		return inserted;
	}
	inline bool is_compleat()
	{
		return inserted >= size;
	}
	void insert(double *input, double *output)
	{
		if (is_compleat())
		{
			printf("\nWARNING: Cannot insert, nn_trainingData is Full. NOTE decleared size of nn_trainingData was- %i\nnn_trainingData::insert() Failed\n", get_size());
			return;
		}
		if (input[0] != nn_neuronCount(gradient, 1) + 1 or output[0] != nn_neuronCount(gradient, nn_layerCount(gradient)) + 1)
		{
			printf("\nERROR: Array of incorrect length passed. NOTE decleared nn has %i input neurons and %i output neurons\nnn_trainingData::insert() Aborted\n", nn_neuronCount(gradient, 1), nn_neuronCount(gradient, nn_layerCount(gradient)));
			exit(1);
		}
		inputs[inserted] = input;
		outputs[inserted] = output;
		inserted++;
	}
	void clear()
	{
		inserted = 0;
	}
	~nn_trainingData()
	{
		if (size > 0)
		{
			free(inputs);
			free(outputs);
			nn_delete(gradient);						
		}
	}
};
void nn_decendGradient(double ***nn, nn_trainingData& examples)
{
	if (!examples.is_compleat())
	{
		printf("\nERROR: Incompleat nn_trainingData passed, decleared nn_trainingData size was %i but had only %i elements inserted\nnn_decendGradient() Aborted\n", examples.get_size(), examples.get_insertionCount());
		exit(1);
		return;
	}
	//computing gradient
	nn_clear(examples.gradient);
	for (int example_no = 0; example_no < examples.get_size(); example_no++)
	{
		nn_input(nn, examples.inputs[example_no]);
		nn_process(nn);
		nn_derivate(nn, examples.outputs[example_no], examples.gradient);
	}
	//updating weights
	for (int layer_no = 1; layer_no < nn[0][0][0]; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			for (int weight_no = 2; weight_no < nn[layer_no][neuron_no][0]; weight_no++)
			{
				nn[layer_no][neuron_no][weight_no] -= nn_LEARNING_RATE * examples.gradient[layer_no][neuron_no][weight_no] / examples.get_size();
			}
		}
	}
}
double nn_totalCost(double ***nn, nn_trainingData& examples)
{
	double totalCost = 0;
	for (int example_no = 0; example_no < examples.get_size(); example_no++)
	{
		nn_input(nn, examples.inputs[example_no]);
		nn_process(nn);
		totalCost += nn_cost(nn, examples.outputs[example_no]);
	}
	return totalCost / examples.get_size();
}
void pool_executor(int, nn_thread *);
class nn_thread
{
	friend void pool_executor(int, nn_thread *);
	friend void pool_totalCost(int, nn_thread *);
	friend double nn_totalCost(nn_thread *, nn_trainingData&);
	friend void pool_decendGradient(int, nn_thread *);
	friend void nn_decendGradient(nn_thread *, nn_trainingData&, double);
	friend void pool_cmpt_activation_hidden(int, nn_thread *);
	friend void pool_cmpt_activation_output(int, nn_thread *);
	friend double **nn_process(nn_thread *);
	std::thread *threads;
	std::mutex locker;
	double ****nns;
	std::atomic<bool> alive = 0, invoking = 0;
	std::atomic<char> compleated = 0;
	char func_type;

	//functions for thread pool
	double **pool_input, **pool_output;
	double ****pool_gradient;
	double pool_totalCost_totalCost;
	double **pool_prev_layer;

  public:
	double ***nn;
	char thread_ct;
	nn_thread(double ***arg_nn, int arg_thread_ct)
	{
		nn = arg_nn;
		if (arg_thread_ct <= 0)
		{
			printf("\nWARNING: Cannot run a program with 0 threads so thread_ct was set to 1\nnn_thread::constructor() Invalid argument\n");
			arg_thread_ct = 1;
		}
		thread_ct = arg_thread_ct;
		//Memory allocation
		nns = (double ****)malloc(8 * arg_thread_ct);
		nns[0] = (double ***)malloc(8);
		nns[0][0] = (double **)malloc(8);
		nns[0][0][0] = (double *)malloc(8);
		*nns[0][0][0] = arg_thread_ct;
		threads = new std::thread[thread_ct - 1];
		pool_input = (double **)malloc(8 * thread_ct);
		pool_output = (double **)malloc(8 * thread_ct);
		pool_gradient = (double ****)malloc(8 * thread_ct);
		for (int thread_no = 1; thread_no < thread_ct; thread_no++)
		{
			pool_gradient[thread_no] = nn_dublicate(nn);
			nns[thread_no] = nn_dublicate(nn);
			nn_clear(pool_gradient[thread_no]);
		}
	}
	void update_nns()
	{
		//Copying new values of nn to other nns
		for (int nns_no = 1; nns_no < nns[0][0][0][0]; nns_no++)
		{
			nn_copy(nns[nns_no], nn);
		}
	}
	bool create_threads()
	{
		if (alive)
			return false;
		else
			alive = 1;
		for (int thread_no = 1; thread_no < thread_ct; thread_no++)
		{
			threads[thread_no - 1] = std::thread(pool_executor, thread_no, this);
		}
		return true;
	}
	inline bool executed()
	{
		return (compleated == thread_ct - 1);
	}
	void joinAll()
	{
		while (!executed())
		{
		}
		return;
	}
	inline void input_cmpt_activation(int thread_no, double *neuron)
	{
		pool_input[thread_no] = neuron;
	}
	inline void input_decendGradient(int thread_no, double *input, double *output)
	{
		pool_input[thread_no] = input;
		pool_output[thread_no] = output;
	}
	inline void input_totalCost(int thread_no, double *input, double *output)
	{
		pool_input[thread_no] = input;
		pool_output[thread_no] = output;
	}
	void pool_applyGamma(double gamma)
	{
		for (int thread_no = 1; thread_no < thread_ct; thread_no++)
		{
			nn_applyGamma(pool_gradient[thread_no], gamma);
		}
	}
	inline void invokeAll(char arg_func_type)
	{
		func_type = arg_func_type;
		compleated = 0;
		invoking = invoking ^ 1;
	}
	inline bool kill()
	{
		if (!alive)
			return false;
		alive = 0;
		invoking = 0;
		for (int thread_no = 1; thread_no < thread_ct; thread_no++)
		{
			threads[thread_no - 1].join();
		}
		return true;
	}
	~nn_thread()
	{
		kill();
		delete[] threads;
		free(pool_input);
		free(pool_output);
		for (int thread_no = 1; thread_no < thread_ct; thread_no++)
		{
			nn_delete(pool_gradient[thread_no]);
		}
		free(pool_gradient);
		for (int nns_no = 1; nns_no < ****nns; nns_no++)
		{
			nn_delete(nns[nns_no]);
		}
		free(nns[0][0][0]);
		free(nns[0][0]);
		free(nns[0]);
		free(nns);
	}
};
inline void pool_cmpt_activation_hidden(int thread_no, nn_thread *thread_obj)
{
	double activation = 0;
	int neuron_no = 1;
	//aliasing neuron == pool_input[thread_no];
	for (; neuron_no < **thread_obj->pool_prev_layer; neuron_no++)
	{
		activation += *(*(thread_obj->pool_prev_layer + neuron_no) + 1) * (*((thread_obj->pool_input)[thread_no] + neuron_no + 1)); //Adding weigted sum
	}
	activation += *((thread_obj->pool_input)[thread_no] + neuron_no + 1); //Adding bias
	*((thread_obj->pool_input)[thread_no] + 1) = nn_activationf(activation);
	thread_obj->compleated++;
}
inline void pool_cmpt_activation_output(int thread_no, nn_thread *thread_obj)
{
	double activation = 0;
	int neuron_no = 1;
	//aliasing neuron == pool_input[thread_no];
	for (; neuron_no < **thread_obj->pool_prev_layer; neuron_no++)
	{
		activation += *(*(thread_obj->pool_prev_layer + neuron_no) + 1) * (*((thread_obj->pool_input)[thread_no] + neuron_no + 1)); //Adding weigted sum
	}
	activation += *((thread_obj->pool_input)[thread_no] + neuron_no + 1); //Adding bias
	*((thread_obj->pool_input)[thread_no] + 1) = nn_outputActivationf(activation);
	thread_obj->compleated++;
}
inline void pool_decendGradient(int thread_no, nn_thread *thread_obj)
{
	nn_input((thread_obj->nns)[thread_no], (thread_obj->pool_input)[thread_no]);
	nn_process((thread_obj->nns)[thread_no]);
	nn_derivate((thread_obj->nns)[thread_no], (thread_obj->pool_output)[thread_no], (thread_obj->pool_gradient)[thread_no]);
	thread_obj->compleated++;
}
inline void pool_totalCost(int thread_no, nn_thread *thread_obj)
{
	nn_input((thread_obj->nns)[thread_no], (thread_obj->pool_input)[thread_no]);
	nn_process((thread_obj->nns)[thread_no]);
	double cost = nn_cost((thread_obj->nns)[thread_no], (thread_obj->pool_output)[thread_no]);
	thread_obj->locker.lock();
	thread_obj->pool_totalCost_totalCost +=cost;
	thread_obj->locker.unlock();
	thread_obj->compleated++;
}
void pool_executor(int thread_no, nn_thread *thread_obj)
{
	bool local_invoking = 0;
	while (1)
	{
		while (local_invoking == thread_obj->invoking)
		{
			if (!thread_obj->alive)
				return;
		}

		local_invoking = thread_obj->invoking; //deactivating
		switch (thread_obj->func_type)
		{
		case 0:
			pool_cmpt_activation_hidden(thread_no, thread_obj);
			break;
		case 1:
			pool_cmpt_activation_output(thread_no, thread_obj);
			break;
		case 2:
			pool_decendGradient(thread_no, thread_obj);
			break;
		case 3:
			pool_totalCost(thread_no, thread_obj);
		}
	}
}
double **nn_process(nn_thread *thread_obj)
{
	int layer_no = 2;
	for (; (layer_no + 1) < ***thread_obj->nn; layer_no++)
	{
		thread_obj->pool_prev_layer = *(thread_obj->nn + layer_no - 1);
		int neuron_no = 1;
		while (neuron_no + thread_obj->thread_ct <= ***(thread_obj->nn + layer_no))
		{
			for (int thread_no = 1; thread_no < thread_obj->thread_ct; thread_no++)
			{
				thread_obj->input_cmpt_activation(thread_no, *(*(thread_obj->nn + layer_no) + neuron_no + thread_no));
			}
			thread_obj->invokeAll(0);
			nn_cmpt_activation_hidden(*(thread_obj->nn + layer_no - 1), *(*(thread_obj->nn + layer_no) + neuron_no));
			thread_obj->joinAll();
			neuron_no += thread_obj->thread_ct;
		}
		for (; neuron_no < ***(thread_obj->nn + layer_no); neuron_no++)
			nn_cmpt_activation_hidden(*(thread_obj->nn + layer_no - 1), *(*(thread_obj->nn + layer_no) + neuron_no));
	}
	//Processing for output neurons
	thread_obj->pool_prev_layer = *(thread_obj->nn + layer_no - 1);
	int neuron_no = 1;
	while (neuron_no + thread_obj->thread_ct <= ***(thread_obj->nn + layer_no))
	{
		for (int thread_no = 1; thread_no < thread_obj->thread_ct; thread_no++)
		{
			thread_obj->input_cmpt_activation(thread_no, *(*(thread_obj->nn + layer_no) + neuron_no + thread_no));
		}
		thread_obj->invokeAll(1);
		nn_cmpt_activation_output(*(thread_obj->nn + layer_no - 1), *(*(thread_obj->nn + layer_no) + neuron_no));
		thread_obj->joinAll();
		neuron_no += thread_obj->thread_ct;
	}
	for (; neuron_no < ***(thread_obj->nn + layer_no); neuron_no++)
		nn_cmpt_activation_output(*(thread_obj->nn + layer_no - 1), *(*(thread_obj->nn + layer_no) + neuron_no));
	return *(thread_obj->nn + (int)(***thread_obj->nn) - 1);
}
void nn_decendGradient(nn_thread *thread_obj, nn_trainingData& examples, double gamma)
{
	if (!examples.is_compleat())
	{
		printf("\nERROR: Incompleat nn_trainingData passed, decleared nn_trainingData size was %i but had only %i elements inserted\nnn_decendGradient() Aborted\n", examples.get_size(), examples.get_insertionCount());
		exit(1);
		return;
	}
	//computing gradient
	nn_applyGamma(examples.gradient, gamma);
	thread_obj->pool_applyGamma(gamma);
	int example_no = 0;
	while (example_no + thread_obj->thread_ct <= examples.get_size())
	{
		for (int thread_no = 1; thread_no < thread_obj->thread_ct; thread_no++)
		{
			thread_obj->input_decendGradient(thread_no, examples.inputs[example_no + thread_no], examples.outputs[example_no + thread_no]);
		}
		thread_obj->invokeAll(2);
		nn_input(thread_obj->nn, examples.inputs[example_no]);
		nn_process(thread_obj->nn);
		nn_derivate(thread_obj->nn, examples.outputs[example_no], examples.gradient);
		thread_obj->joinAll();
		example_no += thread_obj->thread_ct;
	}
	for (; example_no < examples.get_size(); example_no++)
	{
		nn_input(thread_obj->nn, examples.inputs[example_no]);
		nn_process(thread_obj->nn);
		nn_derivate(thread_obj->nn, examples.outputs[example_no], examples.gradient);
	}
	//updating weights
	for (int layer_no = 1; layer_no < thread_obj->nn[0][0][0]; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < thread_obj->nn[layer_no][0][0]; neuron_no++)
		{
			for (int weight_no = 2; weight_no < thread_obj->nn[layer_no][neuron_no][0]; weight_no++)
			{
				thread_obj->nn[layer_no][neuron_no][weight_no] -= nn_LEARNING_RATE * examples.gradient[layer_no][neuron_no][weight_no] / examples.get_size();
			}
		}
	}
	for (int thread_no = 1; thread_no < thread_obj->thread_ct; thread_no++)
	{
		for (int layer_no = 1; layer_no < thread_obj->nn[0][0][0]; layer_no++)
		{
			for (int neuron_no = 1; neuron_no < thread_obj->nn[layer_no][0][0]; neuron_no++)
			{
				for (int weight_no = 2; weight_no < thread_obj->nn[layer_no][neuron_no][0]; weight_no++)
				{
					thread_obj->nn[layer_no][neuron_no][weight_no] -= nn_LEARNING_RATE * thread_obj->pool_gradient[thread_no][layer_no][neuron_no][weight_no] / examples.get_size();
				}
			}
		}
	}
}
double nn_totalCost(nn_thread *thread_obj, nn_trainingData &examples)
{
	if (!examples.is_compleat())
	{
		printf("\nERROR: Incompleat nn_trainingData passed, decleared nn_trainingData size was %i but had only %i elements inserted\nnn_decendGradient() Aborted\n", examples.get_size(), examples.get_insertionCount());
		exit(1);
		return -1;
	}
	double cost;
	thread_obj->pool_totalCost_totalCost = 0;
	int example_no = 0;
	while (example_no + thread_obj->thread_ct <= examples.get_size())
	{
		for (int thread_no = 1; thread_no < thread_obj->thread_ct; thread_no++)
		{
			thread_obj->input_totalCost(thread_no, examples.inputs[example_no + thread_no], examples.outputs[example_no + thread_no]);
		}
		thread_obj->invokeAll(3);
		nn_input(thread_obj->nn, examples.inputs[example_no]);
		nn_process(thread_obj->nn);
		cost = nn_cost(thread_obj->nn, examples.outputs[example_no]);
		thread_obj->joinAll();
		thread_obj->pool_totalCost_totalCost +=cost;
		example_no += thread_obj->thread_ct;
	}
	for (; example_no < examples.get_size(); example_no++)
	{
		nn_input(thread_obj->nn, examples.inputs[example_no]);
		nn_process(thread_obj->nn);
		thread_obj->pool_totalCost_totalCost += nn_cost(thread_obj->nn, examples.outputs[example_no]);
	}
	return thread_obj->pool_totalCost_totalCost / examples.get_size();
}