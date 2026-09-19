#include "../include/heat_solver.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

__global__ void heatShared(const float* d_u, float* d_u_tmp, 
                                   int nx, int ny, float cx) {
    
    // Allocation de la mémoire partagée avec la bordure (Halo)
    __shared__ float s_u[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    // Identifiants locaux dans le bloc
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identifiants globaux dans la grille 2D
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Coordonnées décalées dans la mémoire partagée (pour laisser la place au halo)
    int sx = tx + 1;
    int sy = ty + 1;

    // Index linéaire 1D pour la mémoire globale
    int global_idx = y * nx + x;

    // 1. Chargement de la tuile centrale
    if (x < nx && y < ny) {
        s_u[sy][sx] = d_u[global_idx];
        
        // 2. Chargement collaboratif des Halos par les threads situés sur les bords du bloc
        if (tx == 0 && x > 0) 
            s_u[sy][0] = d_u[global_idx - 1]; // Halo Gauche
            
        if (tx == BLOCK_DIM_X - 1 && x < nx - 1) 
            s_u[sy][BLOCK_DIM_X + 1] = d_u[global_idx + 1]; // Halo Droit
            
        if (ty == 0 && y > 0) 
            s_u[0][sx] = d_u[global_idx - nx]; // Halo Haut
            
        if (ty == BLOCK_DIM_Y - 1 && y < ny - 1) 
            s_u[BLOCK_DIM_Y + 1][sx] = d_u[global_idx + nx]; // Halo Bas
    }

    // Synchronisation obligatoire pour s'assurer que toute la tuile et le halo sont chargés
    __syncthreads();

    // 3. Calcul du Stencil 2D à 5 points uniquement pour l'intérieur de la plaque
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
        d_u_tmp[global_idx] = s_u[sy][sx] + cx * (
                              s_u[sy - 1][sx] + s_u[sy + 1][sx] + // Haut et Bas
                              s_u[sy][sx - 1] + s_u[sy][sx + 1] - // Gauche et Droite
                              4.0f * s_u[sy][sx]);                // Centre
    }
}

// Fonction de lancement côté Host
void solveHeatGPUShared2D(float* d_u, float* d_u_tmp, int nx, int ny, 
                          float alpha, float dx, float dt, int steps) {
    float cx = (alpha * dt) / (dx * dx);

    dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 numBlocks((nx + BLOCK_DIM_X - 1) / BLOCK_DIM_X, 
                   (ny + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    for (int t = 0; t < steps; ++t) {
        heatShared2DKernel<<<numBlocks, threadsPerBlock>>>(d_u, d_u_tmp, nx, ny, cx);
        
        float* temp = d_u;
        d_u = d_u_tmp;
        d_u_tmp = temp;
    }
    cudaDeviceSynchronize();
}