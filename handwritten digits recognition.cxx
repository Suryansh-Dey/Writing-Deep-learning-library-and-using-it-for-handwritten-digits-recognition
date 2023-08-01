#include <iostream>
#include "NeuralNetwork2.cxx"
inline float rand_float()
{
	return (2 * (float(rand()) / INT_MAX) - 1);
}
inline int block_positive(int move)
{
	if (move > 0)
		return 0;
	else
		return move;
}
inline int block_negetive(int move)
{
	if (move < 0)
		return 0;
	else
		return move;
}
inline int appx(float val)
{
	return (val < 0) * (val - 0.5) + (val >= 0) * (val + 0.5);
}
void scale(double gray_vals[784], double enlargement)
{
	if (enlargement < 1)
		goto ScallingDown;
	//Horizontal Scalling
	for (int pixel_x = 0; pixel_x < 14; pixel_x++)
	{
		for (int pixel_y = 0; pixel_y < 28; pixel_y++)
		{
			gray_vals[pixel_x + pixel_y * 28] = gray_vals[14 - appx((14 - pixel_x) / enlargement) + pixel_y * 28];
		}
	}
	for (int pixel_x = 28 - 1; pixel_x >= 14; pixel_x--)
	{
		for (int pixel_y = 0; pixel_y < 28; pixel_y++)
		{
			gray_vals[pixel_x + pixel_y * 28] = gray_vals[14 - appx((14 - pixel_x) / enlargement) + pixel_y * 28];
		}
	}
	//Vertical Scalling
	for (int pixel_x = 0; pixel_x < 28; pixel_x++)
	{
		for (int pixel_y = 0; pixel_y < 14; pixel_y++)
		{
			gray_vals[pixel_x + pixel_y * 28] = gray_vals[(14 - appx((14 - pixel_y) / enlargement)) * 28 + pixel_x];
		}
	}
	for (int pixel_x = 0; pixel_x < 28; pixel_x++)
	{
		for (int pixel_y = 28 - 1; pixel_y >= 14; pixel_y--)
		{
			gray_vals[pixel_x + pixel_y * 28] = gray_vals[(14 - appx((14 - pixel_y) / enlargement)) * 28 + pixel_x];
		}
	}
	return;
ScallingDown:
	for (int pixel_x = 14 - 1; pixel_x >= 0; pixel_x--)
	{
		for (int pixel_y = 0; pixel_y < 28; pixel_y++)
		{
			if (14 - int((14 - pixel_x) / enlargement + 0.5) >= 0)
				gray_vals[pixel_x + pixel_y * 28] = gray_vals[14 - appx((14 - pixel_x) / enlargement) + pixel_y * 28];
			else
				gray_vals[pixel_x + pixel_y * 28] = 0;
		}
	}
	for (int pixel_x = 14; pixel_x < 28; pixel_x++)
	{
		for (int pixel_y = 0; pixel_y < 28; pixel_y++)
		{
			if (14 - int((14 - pixel_x) / enlargement + 0.5) < 28)
				gray_vals[pixel_x + pixel_y * 28] = gray_vals[14 - appx((14 - pixel_x) / enlargement) + pixel_y * 28];
			else
				gray_vals[pixel_x + pixel_y * 28] = 0;
		}
	}
	//Vertical Scalling
	for (int pixel_x = 0; pixel_x < 28; pixel_x++)
	{
		for (int pixel_y = 14 - 1; pixel_y >= 0; pixel_y--)
		{
			if (14 - appx((14 - pixel_y) / enlargement) >= 0)
				gray_vals[pixel_x + pixel_y * 28] = gray_vals[(14 - appx((14 - pixel_y) / enlargement)) * 28 + pixel_x];
			else
				gray_vals[pixel_x + pixel_y * 28] = 0;
		}
	}
	for (int pixel_x = 0; pixel_x < 28; pixel_x++)
	{
		for (int pixel_y = 14; pixel_y < 28; pixel_y++)
		{
			if (14 - appx((14 - pixel_y) / enlargement) < 28)
				gray_vals[pixel_x + pixel_y * 28] = gray_vals[(14 - appx((14 - pixel_y) / enlargement)) * 28 + pixel_x];
			else
				gray_vals[pixel_x + pixel_y * 28] = 0;
		}
	}
}
constexpr int BATCH_SIZE = 200, TOTAL_BATCHES = 30, MAX_MOVE = 4;
constexpr float MAX_SCALE = 0.5;
constexpr int EPOCH = 100;
constexpr float BASE_LEARINING_RATE = 0.1;
double gray_vals[BATCH_SIZE][784 + 1];
double lable_vals[BATCH_SIZE][10 + 1];
int main(int argc, char *argv[])
{
	std::ifstream images_file("train-images.idx3-ubyte", std::ios::binary);
	std::ifstream lables_file("train-labels.idx1-ubyte", std::ios::binary);
	//Creating a Artificial neural network
	//int neuron_counts[] = {4, 784, 20, 10};
	//double ***model = nn_create(neuron_counts);
	//nn_mutate(model, 1, 0.5); // Initialising with random weights
	double ***model = nn_load("error.txt");
	nn_set_learningRate(0.1);
	nn_thread model_threads(model, 4);
	model_threads.create_threads();
	for (int epoch_no = 0; epoch_no < EPOCH; epoch_no++)
	{
		images_file.seekg(16);
		lables_file.seekg(8);
		for (int batch_no = 1; batch_no < TOTAL_BATCHES + 1; batch_no++)
		{
			//Copying images_file data to gray_vals
			char pixel_byte;
			for (int image_no = 0; image_no < BATCH_SIZE; image_no++)
			{
				int move_x = int(MAX_MOVE * rand_float() + 0.1), move_y = int(MAX_MOVE * rand_float() + 0.1);
				int move = 28 * move_y + move_x;
				if (move > 0)
				{
					for (int skip_no = 1; skip_no < move + 1; skip_no++)
					{
						gray_vals[image_no][skip_no] = 0;
					}
				}
				else
				{
					for (int skip_no = 1; skip_no < abs(move) + 1; skip_no++)
					{
						images_file.read(&pixel_byte, 1);
					}
				}
				gray_vals[image_no][0] = 784 + 1;
				for (int pixel_no = block_negetive(move) + 1; pixel_no < 784 + 1 + block_positive(move); pixel_no++)
				{
					images_file.read(&pixel_byte, 1);
					gray_vals[image_no][pixel_no] = double(pixel_byte) / 255;
				}
				if (move > 0)
				{
					for (int skip_no = 1; skip_no < move + 1; skip_no++)
					{
						images_file.read(&pixel_byte, 1);
					}
				}
				else
				{
					for (int remaining_no = 784 + 1 + move; remaining_no < 784 + 1; remaining_no++)
					{
						gray_vals[image_no][remaining_no] = 0;
					}
				}

				scale(gray_vals[image_no] + 1, 1 + rand_float() * MAX_SCALE / (fmax(abs(move_x), abs(move_y)) + 1));
			}
			//Copying lables_file data to lable_vals
			char lable_val;
			for (int lable_no = 0; lable_no < BATCH_SIZE; lable_no++)
			{
				lables_file.read(&lable_val, 1);
				lable_vals[lable_no][0] = 10 + 1;
				for (int byte_idx = 1; byte_idx < 10 + 1; byte_idx++)
				{
					if (lable_val == byte_idx - 1)
						lable_vals[lable_no][byte_idx] = 1;
					else
						lable_vals[lable_no][byte_idx] = 0;
				}
			}
			//Inserting inputs and outputs to trainingData
			nn_trainingData examples(model, BATCH_SIZE);
			for (int image_no = 0; image_no < BATCH_SIZE; image_no++)
			{
				examples.insert(*(gray_vals + image_no), *(lable_vals + image_no));
			}
			//Running an Epoch of gradient decend algorithm
			nn_decendGradient(&model_threads, examples, 0.5);
			model_threads.update_nns();
			//std::cout << nn_totalCost(&model_threads, examples) << '\n';
			examples.clear();
		}
		std::cout << "epoch no = " << epoch_no + 1 << '\n';
	}
	nn_save(model, "error.txt");
	lables_file.close();
	images_file.close();
	return 0;
}