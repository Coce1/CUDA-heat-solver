#include "../include/heat_solver.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

__global__ void heatShared(const int* row_ptr, const int* col_idx, const float* val, 
                                       const float* u_n, float* u_next, float cx) {
    
    // NOUVELLE LOGIQUE : 1 Bloc de threads s'occupe d'UNE SEULE ligne entière
    int row = blockIdx.x; 
    int tid = threadIdx.x;

    // Allocation de la mémoire partagée pour la réduction
    extern __shared__ float s_partial_sums[];

    float local_sum = 0.0f;
    int row_start = row_ptr[row];
    int row_end   = row_ptr[row + 1];

    // 1. Les threads du bloc se partagent le travail de la ligne
    // Si la ligne a 5 éléments et le bloc a 32 threads, seuls les 5 premiers travaillent.
    for (int i = row_start + tid; i < row_end; i += blockDim.x) {
        local_sum += val[i] * u_n[col_idx[i]];
    }
    
    // Dépôt dans le cache partagé
    s_partial_sums[tid] = local_sum;
    __syncthreads();

    // 2. Arbre de réduction en mémoire partagée (Évite les conflits de banque)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial_sums[tid] += s_partial_sums[tid + stride];
        }
        __syncthreads();
    }

    // 3. Le premier thread (le chef d'équipe) écrit le résultat physique final
    if (tid == 0) {
        u_next[row] = u_n[row] + cx * s_partial_sums[0];
    }
}


void solveHeatGPUSharedMatrix(int num_points, const int* d_row_ptr, const int* d_col_idx, 
                              const float* d_val, float* d_u, float* d_u_tmp, 
                              float alpha, float dx, float dt, int steps) {
    
    // Calcul de la constante thermique
    float cx = (alpha * dt) / (dx * dx);

    // Configuration d'exécution : 1 BLOC par point physique (par ligne de la matrice)
    int threadsPerBlock = 32; // Une équipe de 32 threads (un Warp)
    int numBlocks = num_points; 
    
    // Allocation dynamique du cache partagé (32 floats par bloc)
    size_t shared_mem_size = threadsPerBlock * sizeof(float);

    // Boucle temporelle sur le Host (CPU)
    for (int t = 0; t < steps; ++t) {
        
        // Lancement du kernel avec le 3ème argument d'exécution (shared memory)
        heatShared<<<numBlocks, threadsPerBlock, shared_mem_size>>>(d_row_ptr, 
                                                                                d_col_idx, d_val, 
                                                                                d_u, d_u_tmp, cx);

        // Échange des pointeurs (Double Buffering)
        float* temp = d_u;
        d_u = d_u_tmp;
        d_u_tmp = temp;
    }

    // Attente de la fin de tous les pas de temps
    cudaDeviceSynchronize();
}