#include "../include/heat_solver.cuh"
#include <iostream>

// 2D Heat Equation Solver (Finite Difference Method) - CPU Sequential
void solveHeatCPU(float* h_u, float* h_u_tmp, int nx, int ny, 
                  float alpha, float dx, float dy, float dt, int steps) {
                  
    // Calculate the CFL constants for X and Y dimensions
    float cx = (alpha * dt) / (dx * dx);
    float cy = (alpha * dt) / (dy * dy);

    // CFL Stability check: cx + cy must be <= 0.5 for 2D explicit method
    if (cx + cy > 0.5f) {
        std::cerr << "[Warning] CFL condition violated! Simulation may explode." << std::endl;
    }

    // Time-stepping loop
    for (int t = 0; t < steps; ++t) {
        
        // Spatial loop (excluding boundaries: x=0, x=nx-1, y=0, y=ny-1)
        // We leave the borders untouched to enforce Dirichlet boundary conditions (fixed temperature)
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                
                // 1D indexing for a 2D grid
                int idx = y * nx + x;
                
                // Stencil neighbor indices
                int top    = (y - 1) * nx + x;
                int bottom = (y + 1) * nx + x;
                int left   = y * nx + (x - 1);
                int right  = y * nx + (x + 1);

                // Apply the 5-point finite difference stencil
                h_u_tmp[idx] = h_u[idx] + 
                               cx * (h_u[right] + h_u[left] - 2.0f * h_u[idx]) + 
                               cy * (h_u[top] + h_u[bottom] - 2.0f * h_u[idx]);
            }
        }

        // Copy the updated interior domain back to the main grid for the next iteration
        // (In a highly optimized CPU code, we would just swap the pointers instead of copying)
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                int idx = y * nx + x;
                h_u[idx] = h_u_tmp[idx];
            }
        }
    }
}