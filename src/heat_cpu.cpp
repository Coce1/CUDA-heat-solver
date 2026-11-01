#include "../include/heat_solver.cuh"

#include <utility>

float* solveHeatCPU(int num_points, const int* row_ptr, const int* col_idx,
                    const float* val, float* u, float* u_tmp,
                    float alpha, float dx, float dt, int steps) {

    const float cx = heatCoefficient(alpha, dx, dt);

    for (int t = 0; t < steps; ++t) {
        for (int row = 0; row < num_points; ++row) {
            float dot_product = 0.0f;
            const int row_start = row_ptr[row];
            const int row_end = row_ptr[row + 1];

            for (int i = row_start; i < row_end; ++i) {
                dot_product += val[i] * u[col_idx[i]];
            }
            u_tmp[row] = u[row] + cx * dot_product;
        }
        std::swap(u, u_tmp); // double buffering
    }

    // Après le dernier swap, u pointe sur l'état le plus récent : on le retourne
    // explicitement (l'appelant ne voit pas le swap local).
    return u;
}
