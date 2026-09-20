#ifndef HEAT_SOLVER_CUH
#define HEAT_SOLVER_CUH

// CPU sequential implementation
void solveHeatCPU(int num_points, const int* row_ptr, const int* col_idx,
                  const float* val, float* u, float* u_tmp,
                  float alpha, float dx, float dt, int steps);

// GPU implementation - Global Memory (Naive)
void solveHeatGPUNaive(float* d_u, float* d_u_tmp, int nx, 
                        float alpha, float dx, float dt, int steps);

// GPU implementation - Shared Memory (Tiling with Halos)
void solveHeatGPUSharedMatrix(int num_points, const int* d_row_ptr, const int* d_col_idx, 
                              const float* d_val, float* d_u, float* d_u_tmp, 
                              float alpha, float dx, float dt, int steps);

#endif 