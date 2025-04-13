#include <iostream>
#include <cmath>
#include <omp.h>
#include <SDL2/SDL.h>

#define PI 3.14159265358979323846
#define TERMS 100 
#define COLS 800
#define ROWS 600
#define THREAD_NUM 12
#define RADIUS 100

double sin_values[360], cos_values[360];

// Factorial function
double factorial(int n) {
    if (n == 0 || n == 1) return 1;
    double fact = 1;
    for (int i = 2; i <= n; i++) fact *= i;
    return fact;
}

// approx sin
double sin(double x) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int n = 0; n < TERMS; ++n) {
        double term = pow(-1, n)*pow(x,(2*n)+1)/factorial((2*n)+1);
        sum += term;
    }
    return sum;
}

// approx cos
double cos(double x) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int n = 0; n < TERMS; ++n) {
        double term = pow(-1, n) * pow(x,(2 *n))/factorial(2*n);
        sum += term;
    }
    return sum;
}

// Convert degrees to radians
double convert_to_radians(double degrees) {
    return degrees * PI / 180.0;
}

void display(double x[], double y[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL Initialization failed: " << SDL_GetError() << std::endl;
        return;
    }

    SDL_Window* window = SDL_CreateWindow("OpenMP Circle", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, COLS, ROWS, SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }

    SDL_SetRenderDrawColor(renderer, 0, 102, 0, 255);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

    for (int t = 0; t < 360; ++t) {
        SDL_RenderDrawPoint(renderer, (int)x[t], (int)y[t]);
    }

    SDL_RenderPresent(renderer);
    SDL_Delay(1000);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

int main() {
    int j = COLS / 2, k = ROWS / 2, r = RADIUS;
    double x[360], y[360];

    omp_set_num_threads(THREAD_NUM);

    double start_time = omp_get_wtime();

    // Parallel computation of sin and cos values
    #pragma omp parallel for schedule(dynamic, 5)
    for (int t = 0; t < 360; ++t) {
        double rad = convert_to_radians(t);
        sin_values[t] = sin(rad);
        cos_values[t] = cos(rad);
    }

    // Parallel computation of x and y
    #pragma omp parallel for schedule(dynamic, 5)
    for (int t = 0; t < 360; ++t) {
        x[t] = r * cos_values[t] + j;
        y[t] = r * sin_values[t] + k;
    }

    double end_time = omp_get_wtime();
    std::cout << "Parallel computation time: " << (end_time - start_time) << " seconds\n";

    // Display the circle
    display(x, y);
    
    return 0;
}