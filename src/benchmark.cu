#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "../include/heat_solver.cuh"
#include <cuda_runtime.h>

void generate2DLaplacianCSR(int nx, int ny, 
                            std::vector<int>& row_ptr, 
                            std::vector<int>& col_idx, 
                            std::vector<float>& val) {
    
    int num_points = nx * ny;
    row_ptr.resize(num_points + 1, 0);

    // Réservation de la mémoire pour éviter les réallocations coûteuses (5 connexions max par point)
    col_idx.reserve(num_points * 5);
    val.reserve(num_points * 5);

    int nnz = 0; // Compteur de valeurs non nulles (Number of Non-Zeros)

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int row = y * nx + x;
            
            // Enregistrement de l'index de départ pour cette ligne
            row_ptr[row] = nnz;

            // On applique le Laplacien uniquement à l'intérieur de la plaque
            if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
                
                // 1. Voisin du Haut (Index le plus petit, inséré en premier)
                val.push_back(1.0f);
                col_idx.push_back((y - 1) * nx + x);
                nnz++;

                // 2. Voisin de Gauche
                val.push_back(1.0f);
                col_idx.push_back(y * nx + (x - 1));
                nnz++;

                // 3. Centre (Diagonale principale)
                val.push_back(-4.0f);
                col_idx.push_back(row);
                nnz++;

                // 4. Voisin de Droite
                val.push_back(1.0f);
                col_idx.push_back(y * nx + (x + 1));
                nnz++;

                // 5. Voisin du Bas (Index le plus grand)
                val.push_back(1.0f);
                col_idx.push_back((y + 1) * nx + x);
                nnz++;
            }
            // Les points sur les bords externes ne rentrent pas dans la condition if.
            // Ils n'auront aucun voisin enregistré. Leur produit scalaire fera donc 0.
            // Équation : U_futur = U_actuel + cx * 0 -> Leur température restera constante.
        }
    }
    
    // Clôture du tableau row_ptr avec le nombre total d'éléments non nuls
    row_ptr[num_points] = nnz; 
}

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// Inclusion de ton fichier d'en-tête contenant les signatures des wrappers
#include "../include/heat_solver.cuh"

int main() {
    // ---------------------------------------------------------
    // PARAMÈTRES PHYSIQUES ET DE SIMULATION
    // ---------------------------------------------------------
    int nx = 1024, ny = 1024;
    int num_points = nx * ny;
    int steps = 1000;

    float alpha = 0.01f;
    float dx = 0.01f;
    float dt = 0.002f;
    
    std::cout << "=== SpMV Heat Equation Benchmark ===" << std::endl;
    std::cout << "Grid: " << nx << "x" << ny << " | Steps: " << steps << "\n" << std::endl;

    // ---------------------------------------------------------
    // 1. CONSTRUCTION DE LA MATRICE CSR (SUR CPU)
    // ---------------------------------------------------------
    std::vector<int> h_row_ptr, h_col_idx;
    std::vector<float> h_val;
    
    generate2DLaplacianCSR(nx, ny, h_row_ptr, h_col_idx, h_val);
    int nnz = h_val.size();
    std::cout << "Matrice CSR générée : " << nnz << " connexions non nulles." << std::endl;

    // ---------------------------------------------------------
    // 2. INITIALISATION THERMIQUE
    // ---------------------------------------------------------
    std::vector<float> h_u(num_points, 20.0f);
    std::vector<float> h_u_tmp(num_points, 20.0f);
    
    // Application d'un choc thermique intense au centre de la plaque
    h_u[(ny / 2) * nx + (nx / 2)] = 1000.0f; 
    h_u_tmp = h_u;

    // Copies indépendantes pour ne pas fausser les tests
    std::vector<float> h_u_cpu = h_u;
    std::vector<float> h_u_cpu_tmp = h_u_tmp;

    // ---------------------------------------------------------
    // 3. ALLOCATION MÉMOIRE GPU (VRAM)
    // ---------------------------------------------------------
    int *d_row_ptr, *d_col_idx;
    float *d_val, *d_u, *d_u_tmp;
    
    cudaMalloc(&d_row_ptr, (num_points + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_val, nnz * sizeof(float));
    cudaMalloc(&d_u, num_points * sizeof(float));
    cudaMalloc(&d_u_tmp, num_points * sizeof(float));

    // Envoi de la matrice (qui est statique géométriquement) vers le GPU une seule fois
    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (num_points + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_val.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);

    // =========================================================
    // EXECUTION ET CHRONOMÉTRAGE
    // =========================================================

    // --- BENCHMARK CPU ---
    std::cout << "\n[CPU] Lancement séquentiel CSR..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    solveHeatCPU(num_points, h_row_ptr.data(), h_col_idx.data(), h_val.data(), 
                 h_u_cpu.data(), h_u_cpu_tmp.data(), alpha, dx, dt, steps);
                 
    double ms_cpu = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - cpu_start).count() * 1000.0;
    std::cout << "-> Temps CPU : " << ms_cpu << " ms" << std::endl;

    // --- BENCHMARK GPU NAÏF ---
    std::cout << "\n[GPU] Lancement Naïf (Global Memory)..." << std::endl;
    // On réinitialise la température sur le GPU avant de lancer le test
    cudaMemcpy(d_u, h_u.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_tmp, h_u_tmp.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
    
    auto gpu_naive_start = std::chrono::high_resolution_clock::now();
    
    solveHeatGPUNaive(num_points, d_row_ptr, d_col_idx, d_val, 
                            d_u, d_u_tmp, alpha, dx, dt, steps);
                            
    double ms_naive = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - gpu_naive_start).count() * 1000.0;
    std::cout << "-> Temps GPU Naïf : " << ms_naive << " ms" << std::endl;

    // --- BENCHMARK GPU OPTIMISÉ ---
    std::cout << "\n[GPU] Lancement Optimisé (Shared Memory / Réduction)..." << std::endl;
    // On réinitialise à nouveau la température pour repartir à zéro
    cudaMemcpy(d_u, h_u.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_tmp, h_u_tmp.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
    
    auto gpu_shared_start = std::chrono::high_resolution_clock::now();
    
    solveHeatGPUSharedMatrix(num_points, d_row_ptr, d_col_idx, d_val, 
                             d_u, d_u_tmp, alpha, dx, dt, steps);
                             
    double ms_shared = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - gpu_shared_start).count() * 1000.0;
    std::cout << "-> Temps GPU Optimisé : " << ms_shared << " ms" << std::endl;

    // ---------------------------------------------------------
    // NETTOYAGE DE LA MÉMOIRE GPU
    // ---------------------------------------------------------
    cudaFree(d_row_ptr); 
    cudaFree(d_col_idx); 
    cudaFree(d_val); 
    cudaFree(d_u); 
    cudaFree(d_u_tmp);

    std::cout << "\nBenchmark terminé avec succès." << std::endl;
    return 0;
}