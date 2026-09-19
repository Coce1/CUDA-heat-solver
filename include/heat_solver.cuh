#ifndef HEAT_SOLVER_CUH
#define HEAT_SOLVER_CUH

// CPU sequential implementation
void solveHeatCPU(float* h_u, float* h_u_tmp, int nx, int ny, 
                  float alpha, float dx, float dy, float dt, int steps);

// GPU implementation - Global Memory (Naive)
void solveHeatGPUNaive(float* d_u, float* d_u_tmp, int nx, int ny, 
                       float alpha, float dx, float dy, float dt, int steps);

// GPU implementation - Shared Memory (Tiling with Halos)
void solveHeatGPUShared(float* d_u, float* d_u_tmp, int nx, int ny, 
                        float alpha, float dx, float dy, float dt, int steps);

#endif // HEAT_SOLVER_CUH