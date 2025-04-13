#include <iostream>
#include <cmath>
#include <SDL2/SDL.h>

#define PI 3.14159265358979323846
#define TERMS 100  // Number of terms in Taylor series
#define COLS 800
#define ROWS 600
#define RADIUS 100

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
    
    for (int n = 0; n < TERMS; ++n) {
        double term = pow(-1, n) * pow(x, 2 * n + 1) / factorial(2 * n + 1);
        sum += term;
    }
    
    return sum;
}

// approx cos
double cos(double x) {
    double sum = 0.0;

    for (int n = 0; n < TERMS; ++n) {
        double term = pow(-1, n) * pow(x, 2 * n) / factorial(2 * n);
        sum += term;
    }
    
    return sum;
}

// degrees to radians
double convert_to_radians(double degrees) {
    return degrees * PI / 180.0;
}

void display(double x[], double y[]){
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("OpenMP Circle", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, COLS, ROWS, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    
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
    
    double start_time = clock();
    
    // x(t) and y(t) serially
    for (int t = 0; t < 360; ++t) {
        double rad = convert_to_radians(t);
        x[t] = r * cos(rad) + j;
        y[t] = r * sin(rad) + k;
    }
    
    double end_time = clock();
    std::cout << "Serial computation time: " << (end_time - start_time) / CLOCKS_PER_SEC << " seconds\n";
    
    // Display
    display(x,y);
    
    return 0;
}
