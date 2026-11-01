#include "../include/heat_solver.cuh"
#include "../include/cuda_utils.cuh"

#include <utility>

// Bloc 32x8 : un warp = 32 points contigus d'une même ligne -> lectures/écritures
// de 128 octets coalescées. (Un bloc 16x16 ferait lire 2 demi-lignes par warp.)
constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;

// Stencil SANS matrice : les poids (1, 1, -4, 1, 1) sont connus du kernel.
// Trafic idéal ~8 octets/point/pas (1 lecture + 1 écriture de u), contre ~48 octets
// avec CSR/ELLPACK (5 col_idx + 5 val relus à chaque pas) : c'est le vrai goulot.
__global__ void heatStencilKernel(const float* __restrict__ u,
                                  float* __restrict__ u_next,
                                  int nx, int ny, float cx) {

    const int x = blockIdx.x * TILE_X + threadIdx.x;
    const int y = blockIdx.y * TILE_Y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t i = static_cast<size_t>(y) * nx + x;
    const float c = u[i];

    // Bords de Dirichlet : température inchangée (même comportement que les lignes
    // vides du CSR). Divergence limitée aux warps touchant le bord.
    if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) {
        u_next[i] = c;
        return;
    }

    u_next[i] = c + cx * (u[i - 1] + u[i + 1] + u[i - nx] + u[i + nx] - 4.0f * c);
}

float* solveHeatGPUStencil(int nx, int ny, float* d_u, float* d_u_tmp,
                           float alpha, float dx, float dt, int steps) {

    const float cx = heatCoefficient(alpha, dx, dt);
    const dim3 block(TILE_X, TILE_Y);
    const dim3 grid((nx + TILE_X - 1) / TILE_X, (ny + TILE_Y - 1) / TILE_Y);

    for (int t = 0; t < steps; ++t) {
        heatStencilKernel<<<grid, block>>>(d_u, d_u_tmp, nx, ny, cx);
        CUDA_CHECK_KERNEL();
        std::swap(d_u, d_u_tmp);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return d_u;
}
