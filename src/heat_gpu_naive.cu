#include "../include/heat_solver.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void heatNaive(const float* d_u, float* d_u_tmp, 
                                  int nx, int ny, float cx) {
    
    // 1. Calcul des coordonnées 2D du thread dans la grille
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Vérification des limites : On ne calcule que l'intérieur de la plaque
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
        
        // 3. Conversion de la coordonnée 2D (x,y) en index 1D linéaire
        int center = y * nx + x;
        
        int top    = (y - 1) * nx + x;
        int bottom = (y + 1) * nx + x;
        int left   = y * nx + (x - 1);
        int right  = y * nx + (x + 1);

        // 4. Stencil 2D à 5 points
        d_u_tmp[center] = d_u[center] + cx * (d_u[top] + d_u[bottom] + 
                                              d_u[left] + d_u[right] - 
                                              4.0f * d_u[center]);
    }
}

void solveHeatGPUNaive(float* d_u, float* d_u_tmp, int nx, 
                         float alpha, float dx, float dt, int steps) {
    
    float cx = (alpha * dt) / (dx * dx);

    // 1D execution configuration
    int threadsPerBlock = 256;
    int numBlocks = (nx + threadsPerBlock - 1) / threadsPerBlock;

    // Time-stepping loop on the Host (CPU)
    for (int t = 0; t < steps; ++t) {
        // Launch kernel
        heatNaive1DKernel<<<numBlocks, threadsPerBlock>>>(d_u, d_u_tmp, nx, cx);
        
        // Pointer swap (Double Buffering)
        float* temp = d_u;
        d_u = d_u_tmp;
        d_u_tmp = temp;
    }
    
    // Wait for the GPU to finish all time steps
    cudaDeviceSynchronize();
}