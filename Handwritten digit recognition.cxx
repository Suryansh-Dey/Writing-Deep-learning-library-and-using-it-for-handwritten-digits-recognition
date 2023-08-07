#include <iostream>
#include "NeuralNetwork2.cxx"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include "Grayscale.cxx"
double gray_vals[784 + 1];
int main(int argc, char *argv[])
{
	double ***model = NN_load("error.txt");
	NN_thread model_threads(model, 4);
	model_threads.create_threads();
	GrayScale board;
	SDL_Texture *digits_img = IMG_LoadTexture(board.s, "digits.png");
	int prediction_display_x = 26 * board.PIXEL_LENGTH, prediction_display_y = 30 * board.PIXEL_LENGTH;
	if (board.LANDSCAPE_MODE)
	{
		prediction_display_x = 30 * board.PIXEL_LENGTH;
		prediction_display_y = 26 * board.PIXEL_LENGTH;
	}
	showNum prediction_display(board.s, digits_img, prediction_display_x, prediction_display_y, 2 * board.PIXEL_LENGTH, 2 * board.PIXEL_LENGTH);
	gray_vals[0] = 784 + 1;
	int maxPrediction_idx;
	while (1)
	{
		board.renderImage(gray_vals + 1);
		board.inputManager(gray_vals + 1);
		NN_input(model, gray_vals);
		double** predictions = NN_process(&model_threads);
		maxPrediction_idx = 1;
		for (int idx = 2; idx < 10 + 1; idx++)
		{
			if (predictions[maxPrediction_idx][1] < predictions[idx][1])
				maxPrediction_idx = idx;
		}
		prediction_display.blit(maxPrediction_idx - 1, predictions[maxPrediction_idx][1] * 255);
		SDL_Delay(50);
	}
	return 0;
}