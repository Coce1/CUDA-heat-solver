#ifndef HEAT_SOLVER_CUH
#define HEAT_SOLVER_CUH

#include <cstdio>
#include <cstdlib>

// Constante thermique cx = alpha*dt/dx^2 du schéma FTCS explicite 2D.
// Stable uniquement si cx <= 0.25 : on refuse de lancer une simulation divergente.
inline float heatCoefficient(float alpha, float dx, float dt) {
    const float cx = (alpha * dt) / (dx * dx);
    if (cx > 0.25f) {
        std::fprintf(stderr, "[heat] schéma FTCS instable : cx = %f > 0.25\n", cx);
        std::exit(EXIT_FAILURE);
    }
    return cx;
}

// Toutes les fonctions ci-dessous retournent le pointeur du buffer qui contient
// l'état FINAL (le double buffering échange u et u_tmp : l'appelant ne peut pas
// deviner lequel des deux contient le résultat).

// CPU séquentiel (référence), matrice CSR.
float* solveHeatCPU(int num_points, const int* row_ptr, const int* col_idx,
                    const float* val, float* u, float* u_tmp,
                    float alpha, float dx, float dt, int steps);

// GPU, CSR-scalar : 1 thread = 1 ligne.
float* solveHeatGPUNaive(int num_points, const int* d_row_ptr, const int* d_col_idx,
                         const float* d_val, float* d_u, float* d_u_tmp,
                         float alpha, float dx, float dt, int steps);

// GPU, ELLPACK stocké colonne-major (max_nnz = 5 pour le Laplacien 2D).
float* solveHeatGPUELLPACK(int num_points, int max_nnz,
                           const int* d_col_idx, const float* d_val,
                           float* d_u, float* d_u_tmp,
                           float alpha, float dx, float dt, int steps);

// GPU, stencil 5 points SANS matrice, mémoire globale (grille nx x ny structurée).
float* solveHeatGPUStencil(int nx, int ny, float* d_u, float* d_u_tmp,
                           float alpha, float dx, float dt, int steps);

// GPU, stencil 5 points SANS matrice, tuile + halo en shared memory.
float* solveHeatGPUSharedStencil(int nx, int ny, float* d_u, float* d_u_tmp,
                                 float alpha, float dx, float dt, int steps);

#endif // HEAT_SOLVER_CUH
