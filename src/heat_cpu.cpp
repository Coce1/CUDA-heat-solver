#include "../include/heat_solver.cuh"
#include <iostream>

void heatMatrixCPU(int num_points, const int* row_ptr, const int* col_idx,
    const float* val, const float* u_n, float* u_next, float cx) {

    // 1. Boucle externe : on parcourt chaque point physique (chaque ligne de la matrice)
    for (int row = 0; row < num_points; ++row) {
        float dot_product = 0.0f;

        // On récupère les limites pour lire uniquement les voisins de CE point
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        // 2. Boucle interne : on multiplie les coefficients par la température des voisins
        for (int i = row_start; i < row_end; ++i) {
            dot_product += val[i] * u_n[col_idx[i]];
        }

        // 3. Mise à jour temporelle du point
        u_next[row] = u_n[row] + cx * dot_product;
    }
}