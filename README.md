# CUDA Heat Solver: SpMV (Sparse Matrix-Vector) Implementation

This project provides a high-performance numerical solver for the two-dimensional heat equation. The core innovation of this solver lies in modeling the spatial Laplacian as a sparse matrix, allowing the thermal problem to be solved through a series of Matrix-Vector multiplications (SpMV) accelerated by GPU (CUDA).

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

### Matrix Formulation (SpMV)
The grid topology (the 5-point stencil) is extracted and baked into a sparse matrix $L$ (the discrete Laplacian). Each row of this matrix represents a physical point, and the non-zero values represent its thermal connections with its geometric neighbors (a weight of $-4$ at the center, and $1$ for the North, South, East, and West neighbors).

Calculating the temperature for the next time step then becomes a simple linear algebra operation:

$$U^{n+1} = U^n + c_x (L \cdot U^n)$$

Where $U$ is the column vector containing the temperature of all the points in the grid.

---

## 2. Benchmark and Performance Analysis

The tests were conducted on a physical grid of $1024 \times 1024$ points (1,048,576 nodes) simulated over 1000 time steps. The resulting sparse matrix contains 5,222,420 non-zero values, which strictly corresponds to an average of about 5 connections per point (an extremely sparse matrix).

| Solver / Format | Execution Time (ms) | Speedup (vs CPU) |
| :--- | :--- | :--- |
| **Sequential CPU (CSR)** | 6975.92 | 1.0x |
| **Optimized GPU (Shared Memory / Reduction)** | 2632.66 | 2.6x |
| **Naive GPU (CSR-Scalar)** | 229.966 | 30.3x |
| **Optimized GPU (Coalesced ELLPACK)** | 211.843 | **32.9x** |

### Results Analysis

1. **The Failure of Shared Memory / Reduction**
   The kernel using shared memory and a block-wise reduction tree turns out to be the slowest GPU method (2632.66 ms). This counter-intuitive result is explained by the nature of the stencil: each row of the matrix contains only 5 values. Assigning an entire block (e.g., 32 threads) to process a single row causes massive hardware underutilization (27 idle threads out of 32). Furthermore, the hardware cost of the synchronization barriers (`__syncthreads()`) required for the reduction is vastly greater than the time needed to simply add 5 floats.

2. **The Paradoxical Efficiency of the Naive Kernel (CSR-Scalar)**
   The "naive" approach (1 thread = 1 row = 1 physical point) offers excellent performance (229.966 ms). Without intra-block synchronization, each thread executes its 5-iteration loop independently. This model (CSR-Scalar) avoids the overhead of shared memory, although it still suffers from unaligned memory requests when reading neighbor temperatures.

3. **The Supremacy of the ELLPACK Format (Coalescence)**
   The ELLPACK-based solver is the most efficient (211.843 ms), outperforming the CPU by nearly 33 times. The ELLPACK format (Column-Major storage) forces perfect memory alignment. When the *warp* (group of 32 threads) requests to read the "left neighbor", the 32 memory addresses are physically adjacent in the VRAM. The GPU then performs a coalesced memory read, maximizing hardware bandwidth and eliminating the main bottleneck of SpMV operations.

---

## 3. Project Architecture

The project is modular and strictly separates declarations (Headers), implementations (Sources), and execution logic (Benchmark). 

### Repository Structure

```text
.
├── include/
│   └── heat_solver.cuh        # Kernel and host function signatures
├── src/
│   ├── benchmark.cu           # Main(): Initialization and timing
│   ├── heat_cpu.cpp           # Reference CPU solver
│   ├── heat_gpu_naive.cu      # CUDA CSR Implementation (1 Thread / Row)
│   ├── heat_gpu_shared.cu     # CUDA CSR Implementation (Block Reduction)
│   └── heat_gpu_ellpack.cu    # CUDA ELLPACK Format Implementation
└── Makefile                   # Build automation
```
## 4. Prerequisites and Execution

Prerequisites
To compile and run this project, your environment must meet the following requirements:

An NVIDIA GPU with CUDA architecture support.

CUDA Toolkit installed (providing the nvcc compiler).

A C++ compiler supporting the C++14 standard minimum (e.g., g++ or clang).

make build automation tool installed on your system.

# Clean previous builds and compile the project with maximum optimization (-O3)
make rebuild

# Run the simulation and display the execution times
./bin/heat_solver
