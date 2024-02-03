#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
class GrayScale
{
  private:
	SDL_Window *window;
	SDL_Texture *white_img;
	SDL_Rect pixel_rect;
	SDL_Event ev;
	double current_fingerPos_x = -1, current_fingerPos_y = -1, last_fingerPos_x = -1, last_fingerPos_y = -1;
	bool writing = 1;
	void renderRect(int x, int y, double gray_val, int width, int height) //alpha is opacity
	{
		pixel_rect.x = x;
		pixel_rect.y = y;
		pixel_rect.w = width;
		pixel_rect.h = height;
		SDL_SetRenderDrawColor(s, gray_val, gray_val, gray_val, 255);
		SDL_RenderFillRect(s, &pixel_rect);
	}

  public:
	SDL_Renderer *s;
	int SCREEN_WIDTH, SCREEN_HEIGHT, PIXEL_LENGTH;
	bool LANDSCAPE_MODE;
	GrayScale()
	{
		//******INITIALISING*******
		SDL_Init(SDL_INIT_EVERYTHING);
		SDL_Window *temp_window = SDL_CreateWindow("SCANNING DISPLAY DIMENTION", 0, 0, 0, 0, SDL_WINDOW_SHOWN);
		SDL_GetWindowSize(temp_window, &SCREEN_WIDTH, &SCREEN_HEIGHT);
		SDL_DestroyWindow(temp_window);
		window = SDL_CreateWindow("INVERSION", 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, 0);
		s = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
		white_img = IMG_LoadTexture(s, "white.png");
		if (SCREEN_WIDTH > SCREEN_HEIGHT)
		{
			PIXEL_LENGTH = SCREEN_HEIGHT / 28;
			LANDSCAPE_MODE = 1;
		}
		else
		{
			LANDSCAPE_MODE = 0;
			PIXEL_LENGTH = SCREEN_WIDTH / 28;
		}
		SDL_RenderClear(s);
	}
	void renderImage(double *gray_vals)
	{
		if (LANDSCAPE_MODE)
		{
			renderRect(29 * PIXEL_LENGTH, 0, 255, 2 * PIXEL_LENGTH, 2 * PIXEL_LENGTH);
			renderRect(29 * PIXEL_LENGTH, 3 * PIXEL_LENGTH, 255, 2 * PIXEL_LENGTH, 2 * PIXEL_LENGTH);
		}
		else
		{
			renderRect(0, 29 * PIXEL_LENGTH, 255, 2 * PIXEL_LENGTH, 2 * PIXEL_LENGTH);
			renderRect(3 * PIXEL_LENGTH, 29 * PIXEL_LENGTH, 255, 2 * PIXEL_LENGTH, 2 * PIXEL_LENGTH);
		}
		for (int y = 0; y < 28; y++)
		{
			for (int x = 0; x < 28; x++)
			{
				if (gray_vals[y * 28 + x] > 1)
					gray_vals[y * 28 + x] = 1;
				renderRect(x * PIXEL_LENGTH, y * PIXEL_LENGTH, 255 * gray_vals[y * 28 + x], PIXEL_LENGTH, PIXEL_LENGTH);
			}
		}
		SDL_SetRenderDrawColor(s, 0, 0, 0, 255);
		SDL_RenderPresent(s);
		SDL_RenderClear(s);
	}
	void captureBoard(double *board_gray_vals)
	{
		if ((current_fingerPos_x > 0) && (current_fingerPos_y > 0) && (current_fingerPos_x < PIXEL_LENGTH * 28) && (current_fingerPos_y < PIXEL_LENGTH * 28))
		{
			if (!writing)
				board_gray_vals[int(current_fingerPos_y / PIXEL_LENGTH) * 28 + int(current_fingerPos_x / PIXEL_LENGTH)] = 0;
			else
			{
				float pixel_gray_val = (2 * float(int(current_fingerPos_y) % PIXEL_LENGTH) / PIXEL_LENGTH) - 1;
				board_gray_vals[int(current_fingerPos_y / PIXEL_LENGTH) * 28 + int(current_fingerPos_x / PIXEL_LENGTH)] += 1 - abs(pixel_gray_val);
				if (pixel_gray_val > 0)
					board_gray_vals[(int(current_fingerPos_y / PIXEL_LENGTH) + 1) * 28 + int(current_fingerPos_x / PIXEL_LENGTH)] += abs(pixel_gray_val);
				if (pixel_gray_val < 0)
					board_gray_vals[(int(current_fingerPos_y / PIXEL_LENGTH) - 1) * 28 + int(current_fingerPos_x / PIXEL_LENGTH)] += abs(pixel_gray_val);

				pixel_gray_val = (2 * float(int(current_fingerPos_x) % PIXEL_LENGTH) / PIXEL_LENGTH) - 1;
				board_gray_vals[int(current_fingerPos_y / PIXEL_LENGTH) * 28 + int(current_fingerPos_x / PIXEL_LENGTH)] += 1 - abs(pixel_gray_val);
				if (pixel_gray_val > 0)
					board_gray_vals[int(current_fingerPos_y / PIXEL_LENGTH) * 28 + int(current_fingerPos_x / PIXEL_LENGTH) + 1] += abs(pixel_gray_val);
				if (pixel_gray_val < 0)
					board_gray_vals[int(current_fingerPos_y / PIXEL_LENGTH) * 28 + int(current_fingerPos_x / PIXEL_LENGTH) - 1] += abs(pixel_gray_val);
			}
		}
	}
	void inputManager(double *gray_vals)
	{
		while (SDL_PollEvent(&ev))
		{
			if (LANDSCAPE_MODE)
			{
				if (ev.type == SDL_FINGERDOWN)
				{
					if ((abs(SCREEN_WIDTH * ev.tfinger.x - 30 * PIXEL_LENGTH) < PIXEL_LENGTH) && (SCREEN_HEIGHT * ev.tfinger.y < 2 * PIXEL_LENGTH))
					{
						writing = writing ^ 1;
						continue;
					}
					if ((abs(SCREEN_WIDTH * ev.tfinger.x - 30 * PIXEL_LENGTH) < PIXEL_LENGTH) && (abs(SCREEN_HEIGHT * ev.tfinger.y - 4 * PIXEL_LENGTH) < 2 * PIXEL_LENGTH))
					{
						for (int idx = 1; idx < 784 + 1; idx++)
						{
							gray_vals[idx] = 0;
						}
						continue;
					}
				}
			}
			else
			{
				if (ev.type == SDL_FINGERDOWN)
				{
					if ((abs(SCREEN_HEIGHT * ev.tfinger.y - 30 * PIXEL_LENGTH) < PIXEL_LENGTH) && (SCREEN_WIDTH * ev.tfinger.x < 2 * PIXEL_LENGTH))
					{
						writing = writing ^ 1;
						continue;
					}
					if ((abs(SCREEN_HEIGHT * ev.tfinger.y - 30 * PIXEL_LENGTH) < PIXEL_LENGTH) && (abs(SCREEN_WIDTH * ev.tfinger.x - 4 * PIXEL_LENGTH) < 2 * PIXEL_LENGTH))
					{
						for (int idx = 1; idx < 784 + 1; idx++)
						{
							gray_vals[idx] = 0;
						}
						continue;
					}
				}
			}
			if (ev.type == SDL_FINGERDOWN)
			{
				current_fingerPos_x = SCREEN_WIDTH * ev.tfinger.x;
				current_fingerPos_y = SCREEN_HEIGHT * ev.tfinger.y;
			}
			if (ev.type == SDL_FINGERUP)
			{
				current_fingerPos_x = -1;
				current_fingerPos_y = -1;
			}
			if (ev.type == SDL_FINGERMOTION)
			{
				current_fingerPos_x += SCREEN_WIDTH * ev.tfinger.dx;
				current_fingerPos_y += SCREEN_HEIGHT * ev.tfinger.dy;
			}
		}
		if ((abs(last_fingerPos_x - current_fingerPos_x) > PIXEL_LENGTH / 2) or (abs(last_fingerPos_y - current_fingerPos_y) > PIXEL_LENGTH / 2) or (!writing))
		{
			last_fingerPos_x = current_fingerPos_x;
			last_fingerPos_y = current_fingerPos_y;
			captureBoard(gray_vals);
		}
	}
};
class showNum
{
	SDL_Renderer *s;
	SDL_Texture *digits_img;
	SDL_Point img_dimention;
	SDL_Rect digit_rect;
	SDL_Rect digit_srcRect;
	int x;
	SDL_Rect rector(int w, int h, int x, int y)
	{
		SDL_Rect rect;
		rect.w = w;
		rect.h = h;
		rect.x = x;
		rect.y = y;
		return rect;
	}

  public:
	void set_rect(int arg_x, int y, int w, int h)
	{
		digit_rect = rector(w, h, arg_x, y);
		x = arg_x;
	}
	showNum(SDL_Renderer *arg_s, SDL_Texture *arg_digits_img, int arg_x, int y, int w = 100, int h = 120)
	{
		s = arg_s;
		digits_img = arg_digits_img;
		SDL_QueryTexture(arg_digits_img, NULL, NULL, &img_dimention.x, &img_dimention.y);
		img_dimention.x /= 10;
		set_rect(arg_x, y, w, h);
	}
	void blit(int num, int alpha = 255)
	{
		SDL_SetTextureAlphaMod(digits_img, alpha);
		float fnum = num;
		if (!num)
		{
			digit_srcRect =
				rector(img_dimention.x, img_dimention.y, 0, 0);
			digit_rect.x = x - digit_rect.w / 2;
			SDL_RenderCopy(s, digits_img, &digit_srcRect, &digit_rect);
			return;
		}
		//converting int to array of int with each digit as it's element
		char digits[11];
		int i = 1;
		for (; num; i++)
		{
			fnum /= 10;
			num /= 10;
			digits[i] = int(0.01 + (fnum - num) * 10);
		}
		digits[0] = i; //storing array length in 0th index

		digit_rect.x = x + digit_rect.w * (digits[0] - 3) / 2;
		for (int i = 1; i < digits[0]; i++)
		{
			digit_srcRect =
				rector(img_dimention.x, img_dimention.y, img_dimention.x * digits[i], 0);
			SDL_RenderCopy(s, digits_img, &digit_srcRect, &digit_rect);
			digit_rect.x -= digit_rect.w + 10;
		}
	}
};