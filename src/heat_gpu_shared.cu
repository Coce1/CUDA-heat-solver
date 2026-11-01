#include "../include/heat_solver.cuh"
#include "../include/cuda_utils.cuh"

#include <utility>

constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;

__device__ __forceinline__ float loadOrZero(const float* __restrict__ u, int nx, int ny,
                                            int gx, int gy) {
    return (gx >= 0 && gx < nx && gy >= 0 && gy < ny)
               ? u[static_cast<size_t>(gy) * nx + gx]
               : 0.0f;
}

// Tuile (TILE_Y+2) x (TILE_X+2) en shared memory : chaque valeur de u est lue UNE fois
// depuis la mémoire globale, puis réutilisée 5 fois (centre + 4 voisins) depuis la shared.
// Le stencil 5 points n'utilise pas les coins du halo : on ne les charge pas.
// Chaque thread charge son propre centre ; seuls les threads de bord de tuile
// chargent en plus un point de halo -> pas de division/modulo, pas de boucle.
// Gain attendu faible sur GPU récent (le L1/L2 absorbe déjà la réutilisation) :
// cette version sert à le mesurer, pas à le supposer.
__global__ void heatStencilSharedKernel(const float* __restrict__ u,
                                        float* __restrict__ u_next,
                                        int nx, int ny, float cx) {

    __shared__ float tile[TILE_Y + 2][TILE_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * TILE_X + tx;
    const int y = blockIdx.y * TILE_Y + ty;

    tile[ty + 1][tx + 1] = loadOrZero(u, nx, ny, x, y);
    if (ty == 0)          tile[0][tx + 1]          = loadOrZero(u, nx, ny, x, y - 1);
    if (ty == TILE_Y - 1) tile[TILE_Y + 1][tx + 1] = loadOrZero(u, nx, ny, x, y + 1);
    if (tx == 0)          tile[ty + 1][0]          = loadOrZero(u, nx, ny, x - 1, y);
    if (tx == TILE_X - 1) tile[ty + 1][TILE_X + 1] = loadOrZero(u, nx, ny, x + 1, y);

    // Barrière OBLIGATOIRE avant toute lecture des voisins écrits par d'autres threads.
    // Elle est placée avant le return de sortie de grille : tous les threads du bloc
    // l'atteignent (une barrière dans une branche divergente = comportement indéfini).
    __syncthreads();

    if (x >= nx || y >= ny) return;

    const size_t i = static_cast<size_t>(y) * nx + x;
    const float c = tile[ty + 1][tx + 1];

    if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) { // Dirichlet
        u_next[i] = c;
        return;
    }

    // Lecture shared : threads consécutifs -> mots consécutifs -> aucun conflit de banque.
    u_next[i] = c + cx * (tile[ty + 1][tx] + tile[ty + 1][tx + 2] +
                          tile[ty][tx + 1] + tile[ty + 2][tx + 1] - 4.0f * c);
}

float* solveHeatGPUSharedStencil(int nx, int ny, float* d_u, float* d_u_tmp,
                                 float alpha, float dx, float dt, int steps) {

    const float cx = heatCoefficient(alpha, dx, dt);
    const dim3 block(TILE_X, TILE_Y);
    const dim3 grid((nx + TILE_X - 1) / TILE_X, (ny + TILE_Y - 1) / TILE_Y);

    for (int t = 0; t < steps; ++t) {
        heatStencilSharedKernel<<<grid, block>>>(d_u, d_u_tmp, nx, ny, cx);
        CUDA_CHECK_KERNEL();
        std::swap(d_u, d_u_tmp);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return d_u;
}
