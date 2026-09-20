#include "../include/heat_solver.cuh"
#include <iostream>


void solveHeatCPU(int num_points, const int* row_ptr, const int* col_idx,
    const float* val, float* u, float* u_tmp,
    float alpha, float dx, float dt, int steps) {

    // Calcul de la constante thermique
    float cx = (alpha * dt) / (dx * dx);

    for (int t = 0; t < steps; ++t) {

        for (int row = 0; row < num_points; ++row) {
            float dot_product = 0.0f;

            int row_start = row_ptr[row];
            int row_end = row_ptr[row + 1];

            for (int i = row_start; i < row_end; ++i) {
                dot_product += val[i] * u[col_idx[i]];
            }

            u_tmp[row] = u[row] + cx * dot_product;
        }

        // Échange des pointeurs (Double Buffering)
        float* temp = u;
        u = u_tmp;
        u_tmp = temp;
    }
}