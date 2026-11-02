# CUDA Heat Solver: SpMV (Sparse Matrix-Vector) Implementation

This project provides a numerical solver for the two-dimensional heat equation on the GPU (CUDA). The thermal problem is first modeled as a series of sparse Matrix-Vector multiplications (SpMV) using two sparse formats (CSR and ELLPACK). It is then compared against **matrix-free stencil kernels**, which exploit the regular structure of the grid and remove the main memory bottleneck of the SpMV approach.

## 1. Mathematical Modeling

### Continuous Equation
The thermal propagation within an isotropic 2D plate is governed by the following parabolic partial differential equation (PDE):

$$\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

Where $u(x,y,t)$ represents the temperature field and $\alpha$ the thermal diffusivity of the material.

### Discretization (Finite Differences)
By applying an explicit Forward-Time Central-Space (FTCS) scheme, we obtain the following numerical approximation for each grid point with coordinates $(i, j)$:

$$\frac{u_{i,j}^{n+1} - u_{i,j}^n}{\Delta t} = \alpha \frac{u_{i-1,j}^n + u_{i+1,j}^n + u_{i,j-1}^n + u_{i,j+1}^n - 4u_{i,j}^n}{\Delta x^2}$$

Setting the CFL stability constant $c_x = \frac{\alpha \Delta t}{\Delta x^2}$, the update equation can be written as:

$$u_{i,j}^{n+1} = u_{i,j}^n + c_x \left( u_{i-1,j}^n + u_{i+1,j}^n + u_{i,j-1}^n + u_{i,j+1}^n - 4u_{i,j}^n \right)$$

The explicit scheme is stable only if $c_x \le 0.25$. The solver checks this condition (`heatCoefficient()`) and refuses to run a diverging simulation.

### Matrix Formulation (SpMV)
The grid topology (the 5-point stencil) is extracted and baked into a sparse matrix $L$ (the discrete Laplacian). Each row of this matrix represents a physical point, and the non-zero values represent its thermal connections with its geometric neighbors (a weight of $-4$ at the center, and $1$ for the North, South, East, and West neighbors). Boundary points (Dirichlet) have no connection: their temperature stays constant.

Calculating the temperature for the next time step then becomes a simple linear algebra operation:

$$U^{n+1} = U^n + c_x (L \cdot U^n)$$

Where $U$ is the column vector containing the temperature of all the points in the grid.

### Matrix-Free Formulation
On a structured grid, the weights $(1, 1, -4, 1, 1)$ are known in advance, so the matrix does not need to be stored nor read. The stencil kernels apply the update equation directly, which is mathematically identical to the SpMV formulation.

---

## 2. GPU Implementations and Optimizations

| Variant | File | Description |
| :--- | :--- | :--- |
| **CPU (CSR)** | `heat_cpu.cpp` | Sequential reference, also used as ground truth for validation |
| **GPU CSR-Scalar** | `heat_gpu_naive.cu` | 1 thread = 1 matrix row = 1 physical point |
| **GPU ELLPACK** | `heat_gpu_ellpack.cu` | Column-major storage, fixed 5 iterations per thread, loop unrolled through a template |
| **GPU Stencil (global)** | `heat_gpu_stencil.cu` | Matrix-free 5-point stencil, global memory only |
| **GPU Stencil (shared)** | `heat_gpu_shared.cu` | Matrix-free stencil with a (8+2)×(32+2) tile and halo in shared memory |

### Key optimizations
- **The matrix is the real bottleneck.** With CSR or ELLPACK, each point re-reads 5 column indices and 5 values at every time step (about 40 bytes), for only 4 bytes of useful temperature reads and 4 bytes written. The matrix-free stencil brings the ideal traffic down to about 8 bytes per point per step.
- **32×8 thread blocks:** one warp covers 32 contiguous points of the same row (128-byte coalesced accesses).
- **ELLPACK:** column-major layout (coalesced reads of `val` and `col_idx`) and `template<int NNZ>` so that `#pragma unroll` is effective (with a runtime `max_nnz`, the compiler does not unroll).
- **Shared stencil:** each value is read once from global memory and reused 5 times from shared memory. Each thread loads its own center; only the threads on the tile border load the halo (no loop, no integer division). `__syncthreads()` is placed before any early `return`.
- **Explicit result buffer:** every solver returns the pointer to the buffer holding the final state (the double-buffering swap is local to the solver).
- **Index arithmetic in `size_t`** to avoid integer overflow on large grids.

### Removed
The former "Shared Memory / Reduction" kernel (1 block of 32 threads per row of only 5 elements, 27 idle threads out of 32, two barriers to sum 5 floats) has been removed. It was the slowest GPU method by design.

---

## 3. Benchmark and Performance Analysis

### Protocol
- Grid of $1024 \times 1024$ points (1,048,576 nodes), 1000 time steps, $\alpha = 0.01$, $\Delta x = 0.01$, $\Delta t = 0.002$ (so $c_x = 0.2$).
- The CSR matrix contains 5,222,420 non-zero values (interior points only: 1022 × 1022 × 5).
- Initial condition: plate at 20 °C with a 1000 °C thermal shock at the center.
- GPU timing with `cudaEvent`, **2 warm-up runs then the average of 5 runs**. The initial state is reloaded outside the timed region. The CPU reference is a single run (`std::chrono`).
- Each GPU result is read from the buffer **returned** by the solver and compared with the CPU reference (`OK` if max absolute difference < 1e-2).

### Results

**Hardware:** Google Colab | GPU: Tesla T4 (sm_75, 40 SM, 15 GB) | CUDA Toolkit: 13.0 (V13.0.88) | Driver: 580.82.07

| Solver / Format | Execution Time (ms) | Speedup (vs CPU) | Throughput (Mpts/s) | max \|diff\| vs CPU | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential CPU (CSR)** | 7335.110 | 1.0x | 143 | (reference) | — |
| **GPU CSR-Scalar** | 212.636 | 34.5x | 4931 | 0.00e+00 | OK |
| **GPU ELLPACK (coalesced)** | 194.200 | 37.8x | 5399 | 0.00e+00 | OK |
| **GPU Stencil (global memory)** | **37.980** | **193.1x** | **27609** | 1.09e-04 | OK |
| **GPU Stencil (shared memory)** | 48.915 | 150.0x | 21437 | 1.09e-04 | OK |

Total program run time: 11.76 s. *Mpts/s = millions of point updates per second.*

### Results Analysis

1. **Removing the matrix is the real optimization (5.1×).**
   The matrix-free stencil (37.98 ms) is 5.1x faster than ELLPACK (194.2 ms). CSR and ELLPACK re-read about 40 bytes of matrix data per point and per step for only 8 bytes of useful temperature traffic; the stencil moves about 8 bytes in total. The measured gain (5.1x) is close to this traffic ratio (about 6x), which is consistent with a memory-bound problem. In the ideal 8-byte model, the global stencil sustains about 221 GB/s (8.4 GB moved in 38 ms), which is roughly 70 % of the 320 GB/s theoretical peak of the T4.

2. **ELLPACK only gains about 9 % over CSR.**
   Going from CSR-Scalar (212.6 ms) to ELLPACK (194.2 ms) improves coalescing of the matrix reads, but the amount of data read stays the same. Coalescence was therefore not the main bottleneck: the volume of matrix data was. The L1/L2 caches already absorbed most of the access-pattern penalty of CSR.

3. **Shared memory is slower than global memory here (29 %).**
   The shared stencil (48.9 ms) is slower than the global one (38.0 ms). On the T4, the L1/L2 caches already capture the reuse of the 5-point stencil, so the explicit tile only adds work: halo loads handled by border threads (divergent branches), shared-memory stores and loads, and one `__syncthreads()` per step and block. This is the explanation consistent with the measurements; it has not been confirmed with a profiler (Nsight Compute).

4. **Numerical validation.**
   CSR and ELLPACK reproduce the CPU result exactly (difference 0). The stencils differ by 1.09e-4 for temperatures reaching 1000: the neighbor sum is evaluated in another order and fused multiply-adds are used. This is a floating-point rounding effect, not a bug.

---

## 4. Project Architecture

The project is modular and strictly separates declarations (Headers), implementations (Sources), and execution logic (Benchmark).

### Repository Structure

```text
.
├── include/
│   ├── cuda_utils.cuh         # CUDA_CHECK, DeviceBuffer (RAII), benchmarkMs, validation helpers
│   └── heat_solver.cuh        # Host function signatures + FTCS stability check
├── src/
│   ├── benchmark.cu           # Main(): matrix generation, timing, validation, summary table
│   ├── heat_cpu.cpp           # Reference CPU solver (CSR)
│   ├── heat_gpu_naive.cu      # CUDA CSR Implementation (1 Thread / Row)
│   ├── heat_gpu_ellpack.cu    # CUDA ELLPACK Format Implementation
│   ├── heat_gpu_stencil.cu    # CUDA matrix-free stencil (global memory)
│   └── heat_gpu_shared.cu     # CUDA matrix-free stencil (shared memory + halo)
└── Makefile                   # Build automation
```

### Error handling and memory
- Every CUDA runtime call is wrapped in `CUDA_CHECK`, and every kernel launch is followed by `CUDA_CHECK_KERNEL()`.
- GPU memory is managed by `DeviceBuffer<T>` (RAII): no manual `cudaFree`.
- The benchmark output is unbuffered, so partial results remain visible if an error occurs.

## 5. Prerequisites

* An NVIDIA GPU with CUDA support. The Makefile defaults to `sm_75` (Turing, e.g. T4). CUDA Toolkit 13 requires Compute Capability 7.5 or higher; with CUDA 11/12, older architectures (`sm_60` to `sm_70`) are still supported.
* CUDA Toolkit 11.0 or higher (tested with 13.0), providing the `nvcc` compiler.
* A C++ compiler supporting the C++14 standard minimum (e.g., g++ or clang).
* GNU Make (optional, see the direct `nvcc` command below).

```bash
nvcc --version
nvidia-smi
```

## 6. Build and Execute

```bash
# Clean previous builds and compile with maximum optimization (-O3)
make rebuild ARCH=sm_75

# Run the simulation and display the execution times
./bin/heat_solver
```

Replace `sm_75` with the architecture of your GPU (`sm_80` A100, `sm_86` RTX 30xx, `sm_89` RTX 40xx, `sm_90` H100).

Equivalent direct command (useful if the Makefile tabs were altered by a copy-paste):

```bash
nvcc -O3 -std=c++14 -arch=sm_75 -lineinfo -Iinclude \
     src/benchmark.cu src/heat_gpu_naive.cu src/heat_gpu_ellpack.cu \
     src/heat_gpu_stencil.cu src/heat_gpu_shared.cu src/heat_cpu.cpp \
     -o heat_solver
./heat_solver
```

### Memory and race checking (optional)
```bash
compute-sanitizer --tool memcheck ./bin/heat_solver
```
This is very slow with 1000 steps: temporarily reduce `steps` to 10 and the grid to 256×256 in `benchmark.cu`.
