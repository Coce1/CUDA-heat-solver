#include "../include/heat_solver.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void heatNaive(int num_points,const int* row_ptr, const int* col_idx, const float* val, 
                                         const float* u_n, float* u_next, float cx) {
    
    // Identifiant global du thread (1 thread = 1 ligne de la matrice = 1 point physique)
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_points) {
        float dot_product = 0.0f;
        
        // On récupère le début et la fin de la ligne dans les vecteurs compressés
        int row_start = row_ptr[row];
        int row_end   = row_ptr[row + 1];

        // Multiplication uniquement avec les voisins existants (les non-zéros)
        for (int i = row_start; i < row_end; ++i) {
            float matrix_val = val[i];             // Ex: -4.0 ou 1.0
            float neighbor_temp = u_n[col_idx[i]]; // Température du voisin
            
            dot_product += matrix_val * neighbor_temp;
        }

        // Application de l'équation : U^{n+1} = U^n + cx * (L * U^n)
        u_next[row] = u_n[row] + cx * dot_product;
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
        heatNaive<<<numBlocks, threadsPerBlock>>>(d_u, d_u_tmp, nx, cx);
        
        // Pointer swap (Double Buffering)
        float* temp = d_u;
        d_u = d_u_tmp;
        d_u_tmp = temp;
    }
    
    // Wait for the GPU to finish all time steps
    cudaDeviceSynchronize();
}

void solveHeatGPUNaiveMatrix(int num_points, const int* d_row_ptr, const int* d_col_idx, 
                             const float* d_val, float* d_u, float* d_u_tmp, 
                             float alpha, float dx, float dt, int steps) {
    
    // Calcul de la constante thermique
    float cx = (alpha * dt) / (dx * dx);

    // Configuration d'exécution : 1 thread par point physique
    int threadsPerBlock = 256;
    int numBlocks = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    // Boucle temporelle sur le Host (CPU)
    for (int t = 0; t < steps; ++t) {
        
        // Lancement du kernel avec les bons paramètres matriciels (CSR)
        heatNaive<<<numBlocks, threadsPerBlock>>>(num_points, d_row_ptr, d_col_idx, d_val, d_u, d_u_tmp, cx);

        // Échange des pointeurs (Double Buffering)
        float* temp = d_u;
        d_u = d_u_tmp;
        d_u_tmp = temp;
    }

    // Attente de la fin de tous les pas de temps
    cudaDeviceSynchronize();
}