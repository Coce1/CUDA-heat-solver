#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

// Vérifie le code retour de tout appel du runtime CUDA et stoppe avec fichier/ligne.
// fflush(stdout) : les chiffres déjà affichés ne sont pas perdus quand une erreur survient.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t cuda_check_err_ = (call);                                  \
        if (cuda_check_err_ != cudaSuccess) {                                  \
            std::fflush(stdout);                                               \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d : %s (%s)\n",             \
                         __FILE__, __LINE__,                                   \
                         cudaGetErrorString(cuda_check_err_), #call);          \
            std::fflush(stderr);                                               \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// À placer juste après un lancement <<<...>>> : détecte les erreurs de configuration
// (grille/bloc invalides, trop de shared memory...). Les erreurs d'exécution
// asynchrones ressortent au prochain cudaDeviceSynchronize()/cudaMemcpy().
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

// Buffer GPU RAII : cudaFree automatique (même en cas de sortie anticipée),
// non copiable, déplaçable. Remplace les cudaMalloc/cudaFree manuels.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        release();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
        size_ = count;
    }

    void release() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    void upload(const T* host, std::size_t count) {
        checkCount(count);
        CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void upload(const std::vector<T>& host) { upload(host.data(), host.size()); }

    void download(T* host, std::size_t count) const {
        checkCount(count);
        CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void download(std::vector<T>& host) const {
        host.resize(size_);
        download(host.data(), size_);
    }

    T* get() const { return ptr_; }
    std::size_t size() const { return size_; }

private:
    void checkCount(std::size_t count) const {
        if (count > size_) {
            std::fflush(stdout);
            std::fprintf(stderr, "[DeviceBuffer] transfert de %zu éléments > capacité %zu\n",
                         count, size_);
            std::exit(EXIT_FAILURE);
        }
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

// Mesure GPU fiable : warm-up (charge les modules/caches, initialise cuBLAS...),
// puis moyenne de `reps` exécutions chronométrées avec cudaEvent.
// `prepare()` (ex. réinitialiser les données) est exécuté HORS de la zone chronométrée.
template <typename Prepare, typename Run>
float benchmarkMs(Prepare&& prepare, Run&& run, int warmup, int reps) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        prepare();
        run();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    for (int i = 0; i < reps; ++i) {
        prepare();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        run();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(reps);
}

// Un NaN (ex. élément jamais écrit par le kernel) est traité comme une erreur infinie.
inline double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (std::isnan(d)) return std::numeric_limits<double>::infinity();
        m = std::max(m, d);
    }
    return m;
}

// Erreur relative |a-b| / max(|b|, 1).
inline double maxRelDiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double ref = static_cast<double>(b[i]);
        const double d = std::fabs(static_cast<double>(a[i]) - ref) / std::max(std::fabs(ref), 1.0);
        if (std::isnan(d)) return std::numeric_limits<double>::infinity();
        m = std::max(m, d);
    }
    return m;
}

inline void printDeviceInfo() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    std::printf("GPU : %s | sm_%d%d | %d SM\n", p.name, p.major, p.minor, p.multiProcessorCount);
}

#endif // CUDA_UTILS_CUH
