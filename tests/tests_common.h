#ifndef CUTANGENT_TESTS_COMMON_H
#define CUTANGENT_TESTS_COMMON_H

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <source_location>
#include <span>

#include <cuda_runtime.h>

namespace cu::test
{

inline void check(cudaError_t err, const std::source_location &loc = std::source_location::current())
{
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA error in %s at\n%s:%u: %s (%s=%d)\n",
                     loc.function_name(),
                     loc.file_name(),
                     loc.line(),
                     cudaGetErrorString(err),
                     cudaGetErrorName(err),
                     static_cast<int>(err));
        std::abort();
    }
}

namespace device
{
    inline void init(int device = 0, const std::source_location &loc = std::source_location::current())
    {
        check(cudaSetDevice(device), loc);
    }

    inline void reset(const std::source_location &loc = std::source_location::current())
    {
        check(cudaDeviceReset(), loc);
    }

} // namespace device

template<typename T, std::size_t N>
struct array
{
    static constexpr std::size_t n_bytes = N * sizeof(T);

    array(const std::source_location &loc = std::source_location::current())
    {
        check(cudaMalloc(reinterpret_cast<void **>(&ptr), n_bytes), loc);
    }
    ~array() { check(cudaFree(ptr)); }

    array &operator=(const std::span<T, N> &host)
    {
        check(cudaMemcpy(ptr, host.data(), n_bytes, cudaMemcpyHostToDevice));
        return *this;
    }

    operator std::array<T, N>() const
    {
        std::array<T, N> host {};
        check(cudaMemcpy(host.data(), ptr, n_bytes, cudaMemcpyDeviceToHost));
        return host;
    }

    T operator[](std::size_t i) const { return ptr[i]; }

    T *data() { return ptr; };
    const T *data() const { return ptr; };

private:
    T *ptr;
};

} // namespace cu::test

#endif // CUTANGENT_TESTS_COMMON_H
