#include <chrono>
#include <cstdio>
#include <vector>

#include "../include/cuda_utils.cuh"
#include "../include/heat_solver.cuh"

// ---------------------------------------------------------------------------
// Génération des matrices (hôte). Ordre des colonnes croissant : haut, gauche,
// centre, droite, bas. Les points de bord n'ont aucune connexion (Dirichlet).
// ---------------------------------------------------------------------------
static void generate2DLaplacianCSR(int nx, int ny, std::vector<int>& row_ptr,
                                   std::vector<int>& col_idx, std::vector<float>& val) {
    const int num_points = nx * ny;
    row_ptr.assign(num_points + 1, 0);
    col_idx.clear();
    val.clear();
    col_idx.reserve(static_cast<size_t>(num_points) * 5);
    val.reserve(static_cast<size_t>(num_points) * 5);

    int nnz = 0;
    auto add = [&](int col, float w) {
        col_idx.push_back(col);
        val.push_back(w);
        ++nnz;
    };

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const int row = y * nx + x;
            row_ptr[row] = nnz;
            if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
                add((y - 1) * nx + x, 1.0f);
                add(y * nx + (x - 1), 1.0f);
                add(row, -4.0f);
                add(y * nx + (x + 1), 1.0f);
                add((y + 1) * nx + x, 1.0f);
            }
        }
    }
    row_ptr[num_points] = nnz;
}

static void generate2DLaplacianELLPACK(int nx, int ny, int max_nnz,
                                       std::vector<int>& col_idx, std::vector<float>& val) {
    const size_t num_points = static_cast<size_t>(nx) * ny;
    col_idx.assign(num_points * max_nnz, 0);
    val.assign(num_points * max_nnz, 0.0f);

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const size_t row = static_cast<size_t>(y) * nx + x;
            const bool interior = (x > 0 && x < nx - 1 && y > 0 && y < ny - 1);

            if (interior) {
                // Colonne-major : indice = n * num_points + row
                val[0 * num_points + row] = 1.0f;  col_idx[0 * num_points + row] = (y - 1) * nx + x;
                val[1 * num_points + row] = 1.0f;  col_idx[1 * num_points + row] = y * nx + (x - 1);
                val[2 * num_points + row] = -4.0f; col_idx[2 * num_points + row] = static_cast<int>(row);
                val[3 * num_points + row] = 1.0f;  col_idx[3 * num_points + row] = y * nx + (x + 1);
                val[4 * num_points + row] = 1.0f;  col_idx[4 * num_points + row] = (y + 1) * nx + x;
            } else {
                // Bord : on pointe sur soi-même avec un poids nul (padding valide, jamais hors bornes)
                for (int n = 0; n < max_nnz; ++n) {
                    col_idx[n * num_points + row] = static_cast<int>(row);
                    val[n * num_points + row] = 0.0f;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Chronomètre + validation d'une variante GPU.
// `run(u, tmp)` doit retourner le pointeur du buffer contenant le résultat.
// ---------------------------------------------------------------------------
template <typename Run>
static void evaluate(const char* name, Run&& run,
                     DeviceBuffer<float>& d_u, DeviceBuffer<float>& d_tmp,
                     const std::vector<float>& h_u0, const std::vector<float>& h_ref,
                     double ms_cpu) {

    // Réinitialisation hors zone chronométrée (les deux buffers = état initial)
    auto prepare = [&]() {
        d_u.upload(h_u0);
        d_tmp.upload(h_u0);
    };

    float* res = nullptr;
    const float ms = benchmarkMs(prepare, [&]() { res = run(d_u.get(), d_tmp.get()); }, 2, 5);

    // Exécution de validation : on lit le buffer RETOURNÉ (pas d_u au hasard)
    prepare();
    res = run(d_u.get(), d_tmp.get());
    std::vector<float> h_gpu(h_u0.size());
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), res, h_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const double diff = maxAbsDiff(h_gpu, h_ref);
    const bool ok = diff < 1e-2; // l'ordre de sommation et les FMA diffèrent du CPU
    std::printf("%-36s : %9.3f ms | x%7.1f vs CPU | max|diff| = %.3e | %s\n",
                name, ms, ms_cpu / ms, diff, ok ? "OK" : "ÉCHEC");
}

int main() {
    // --- Paramètres physiques et de simulation ---
    const int nx = 1024, ny = 1024;
    const int num_points = nx * ny;
    const int steps = 1000;
    const int max_nnz = 5;

    const float alpha = 0.01f;
    const float dx = 0.01f;
    const float dt = 0.002f;

    std::printf("=== SpMV Heat Equation Benchmark ===\n");
    printDeviceInfo();
    std::printf("Grille : %dx%d | Pas de temps : %d\n\n", nx, ny, steps);

    // --- Matrices (hôte) ---
    std::vector<int> h_row_ptr, h_col_idx, h_col_idx_ell;
    std::vector<float> h_val, h_val_ell;
    generate2DLaplacianCSR(nx, ny, h_row_ptr, h_col_idx, h_val);
    generate2DLaplacianELLPACK(nx, ny, max_nnz, h_col_idx_ell, h_val_ell);
    std::printf("Matrice CSR : %zu non-nuls\n\n", h_val.size());

    // --- État initial : plaque à 20 °C avec un choc thermique central ---
    std::vector<float> h_u0(num_points, 20.0f);
    h_u0[(ny / 2) * nx + (nx / 2)] = 1000.0f;

    // --- Référence CPU (une seule exécution, elle sert aussi de vérité terrain) ---
    std::vector<float> cpu_u = h_u0, cpu_tmp = h_u0;
    std::printf("[CPU] référence séquentielle CSR...\n");
    const auto t0 = std::chrono::steady_clock::now();
    const float* cpu_res = solveHeatCPU(num_points, h_row_ptr.data(), h_col_idx.data(), h_val.data(),
                                        cpu_u.data(), cpu_tmp.data(), alpha, dx, dt, steps);
    const double ms_cpu =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const std::vector<float> h_ref(cpu_res, cpu_res + num_points);
    std::printf("CPU CSR séquentiel                   : %9.3f ms\n\n", ms_cpu);

    // --- Mémoire GPU : matrices envoyées UNE fois (RAII : pas de cudaFree manuel) ---
    DeviceBuffer<int> d_row_ptr(h_row_ptr.size()), d_col_idx(h_col_idx.size());
    DeviceBuffer<float> d_val(h_val.size());
    DeviceBuffer<int> d_col_idx_ell(h_col_idx_ell.size());
    DeviceBuffer<float> d_val_ell(h_val_ell.size());
    DeviceBuffer<float> d_u(num_points), d_tmp(num_points);

    d_row_ptr.upload(h_row_ptr);
    d_col_idx.upload(h_col_idx);
    d_val.upload(h_val);
    d_col_idx_ell.upload(h_col_idx_ell);
    d_val_ell.upload(h_val_ell);

    // --- Variantes GPU (warm-up + moyenne sur 5 exécutions chacune) ---
    evaluate("GPU CSR naïf (1 thread/ligne)",
             [&](float* u, float* t) {
                 return solveHeatGPUNaive(num_points, d_row_ptr.get(), d_col_idx.get(), d_val.get(),
                                          u, t, alpha, dx, dt, steps);
             },
             d_u, d_tmp, h_u0, h_ref, ms_cpu);

    evaluate("GPU ELLPACK colonne-major",
             [&](float* u, float* t) {
                 return solveHeatGPUELLPACK(num_points, max_nnz, d_col_idx_ell.get(), d_val_ell.get(),
                                            u, t, alpha, dx, dt, steps);
             },
             d_u, d_tmp, h_u0, h_ref, ms_cpu);

    evaluate("GPU stencil global (sans matrice)",
             [&](float* u, float* t) {
                 return solveHeatGPUStencil(nx, ny, u, t, alpha, dx, dt, steps);
             },
             d_u, d_tmp, h_u0, h_ref, ms_cpu);

    evaluate("GPU stencil shared (tuile + halo)",
             [&](float* u, float* t) {
                 return solveHeatGPUSharedStencil(nx, ny, u, t, alpha, dx, dt, steps);
             },
             d_u, d_tmp, h_u0, h_ref, ms_cpu);

    std::printf("\nBenchmark terminé.\n");
    return 0;
}
