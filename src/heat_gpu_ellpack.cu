

__global__ void heatELLPACKKernel(int num_points, int max_nnz, 
                                  const int* col_idx, const float* val, 
                                  const float* u_n, float* u_next, float cx) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_points) {
        float dot_product = 0.0f;
        
        // Boucle SANS divergence : tous les threads font exactement 5 itérations
        // L'accès mémoire (n * num_points + row) est parfaitement aligné pour le Warp !
        #pragma unroll
        for (int n = 0; n < max_nnz; ++n) {
            int idx = n * num_points + row;
            float matrix_val = val[idx];
            float neighbor_temp = u_n[col_idx[idx]];
            
            dot_product += matrix_val * neighbor_temp;
        }

        u_next[row] = u_n[row] + cx * dot_product;
    }
}


void solveHeatGPUELLPACK(int num_points, int max_nnz, 
                         const int* d_col_idx, const float* d_val, 
                         float* d_u, float* d_u_tmp, 
                         float alpha, float dx, float dt, int steps) {
    
    float cx = (alpha * dt) / (dx * dx);
    int threadsPerBlock = 256;
    int numBlocks = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    for (int t = 0; t < steps; ++t) {
        heatELLPACKKernel<<<numBlocks, threadsPerBlock>>>(num_points, max_nnz, d_col_idx, d_val, d_u, d_u_tmp, cx);

        float* temp = d_u;
        d_u = d_u_tmp;
        d_u_tmp = temp;
    }

    cudaDeviceSynchronize();
}