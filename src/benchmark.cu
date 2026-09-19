#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "../include/heat_solver.cuh"
#include <cuda_runtime.h>

bool verifyResults2D(const std::vector<float>& cpu_res, const std::vector<float>& gpu_res, int nx, int ny) {
    const float tolerance = 1e-4f;
    for (int i = 0; i < nx * ny; ++i) {
        if (std::fabs(cpu_res[i] - gpu_res[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": CPU=" << cpu_res[i] << ", GPU=" << gpu_res[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // 1. Paramètres de la simulation 2D
    int nx = 1024;        // Largeur de la plaque
    int ny = 1024;        // Hauteur de la plaque
    int steps = 1000;     // Nombre de pas de temps

    float alpha = 0.01f;  // Diffusivité thermique
    float dx = 0.01f;     // Pas spatial (identique en x et y)
    float dt = 0.002f;    // Pas de temps (réduit pour la stabilité 2D)

    // Condition CFL pour la 2D (cx <= 0.25)
    float cx = (alpha * dt) / (dx * dx);
    
    std::cout << "=== 2D Heat Equation Solver Benchmark ===" << std::endl;
    std::cout << "Grid Size: " << nx << "x" << ny << " (" << nx * ny << " points) | Time Steps: " << steps << std::endl;
    std::cout << "CFL Constant (cx): " << cx << " (Must be <= 0.25 for 2D stability)\n" << std::endl;

    if (cx > 0.25f) {
        std::cerr << "WARNING: CFL condition violated. Simulation will diverge." << std::endl;
        return -1;
    }

    // 2. Allocation mémoire Host (Tableau 1D aplati)
    size_t bytes = nx * ny * sizeof(float);
    std::vector<float> h_u(nx * ny, 20.0f);      // Température ambiante
    std::vector<float> h_u_tmp(nx * ny, 20.0f);

    // Injection d'un pic de chaleur au centre exact de la plaque 2D
    int center_x = nx / 2;
    int center_y = ny / 2;
    int center_idx = center_y * nx + center_x;
    h_u[center_idx] = 1000.0f;
    h_u_tmp[center_idx] = 1000.0f;

    std::vector<float> h_u_cpu = h_u;
    std::vector<float> h_u_cpu_tmp = h_u_tmp;
    std::vector<float> h_u_gpu_naive(nx * ny, 0.0f);
    std::vector<float> h_u_gpu_shared(nx * ny, 0.0f);

    // 3. Allocation mémoire Device
    float *d_u, *d_u_tmp;
    cudaMalloc(&d_u, bytes);
    cudaMalloc(&d_u_tmp, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ==========================================================
    // BENCHMARK 1: CPU Séquentiel 2D
    // ==========================================================
    std::cout << "[CPU] Running Sequential Solver..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    solveHeatCPU2D(h_u_cpu.data(), h_u_cpu_tmp.data(), nx, ny, alpha, dx, dt, steps);
    
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_stop - cpu_start;
    double ms_cpu = cpu_duration.count() * 1000.0;
    std::cout << "- Time: " << ms_cpu << " ms\n" << std::endl;

    // ==========================================================
    // BENCHMARK 2: GPU Naive 2D (Global Memory)
    // ==========================================================
    std::cout << "[GPU] Running Naive Solver (Global Memory)..." << std::endl;
    cudaMemcpy(d_u, h_u.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_tmp, h_u_tmp.data(), bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    solveHeatGPUNaive2D(d_u, d_u_tmp, nx, ny, alpha, dx, dt, steps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start, stop);
    cudaMemcpy(h_u_gpu_naive.data(), d_u, bytes, cudaMemcpyDeviceToHost);
    
    std::cout << "- Time: " << ms_naive << " ms" << std::endl;
    std::cout << "- Speedup vs CPU: " << std::fixed << std::setprecision(2) << (ms_cpu / ms_naive) << "x\n" << std::endl;

    // ==========================================================
    // BENCHMARK 3: GPU Optimisé 2D (Shared Memory)
    // ==========================================================
    std::cout << "[GPU] Running Optimized Solver (Shared Memory)..." << std::endl;
    cudaMemcpy(d_u, h_u.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_tmp, h_u_tmp.data(), bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    solveHeatGPUShared2D(d_u, d_u_tmp, nx, ny, alpha, dx, dt, steps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_shared = 0;
    cudaEventElapsedTime(&ms_shared, start, stop);
    cudaMemcpy(h_u_gpu_shared.data(), d_u, bytes, cudaMemcpyDeviceToHost);

    std::cout << "- Time: " << ms_shared << " ms" << std::endl;
    std::cout << "- Speedup vs CPU: " << (ms_cpu / ms_shared) << "x" << std::endl;
    std::cout << "- Speedup vs Naive: " << (ms_naive / ms_shared) << "x\n" << std::endl;

    // ==========================================================
    // Validation
    // ==========================================================
    std::cout << "=== Numerical Validation ===" << std::endl;
    bool naive_ok = verifyResults2D(h_u_cpu, h_u_gpu_naive, nx, ny);
    bool shared_ok = verifyResults2D(h_u_cpu, h_u_gpu_shared, nx, ny);
    
    std::cout << "Naive implementation numerical match: " << (naive_ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Shared implementation numerical match: " << (shared_ok ? "PASSED" : "FAILED") << std::endl;

    cudaFree(d_u);
    cudaFree(d_u_tmp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}