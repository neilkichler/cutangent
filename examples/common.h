#ifndef CUTANGENT_COMMON_H
#define CUTANGENT_COMMON_H

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                                \
    do {                                                                             \
        cudaError_t err = x;                                                         \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, \
                    __FILE__, __LINE__, cudaGetErrorString(err),                     \
                    cudaGetErrorName(err), err);                                     \
            abort();                                                                 \
        }                                                                            \
    } while (0)

#endif // CUTANGENT_COMMON_H
