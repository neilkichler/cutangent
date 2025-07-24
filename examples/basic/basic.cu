#include "../common.h"

#include <cuda_runtime.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <cstdio>
#include <iostream>

using T = cu::tangent<double>;

constexpr int n = 36;

// the computation does not make any sense and is just here to show all the operations
constexpr auto f(int out_i, auto x, auto y)
{
    auto print = [out_i](auto x) { printf("[%2d] {%g, %g}\n", out_i, x.v, x.d); };

    int i = 0;
    T vs[n];

    vs[i++] = x + y;
    vs[i++] = x - y;
    vs[i++] = x * y;
    vs[i++] = x / y;
    vs[i++] = max(x, y);
    vs[i++] = min(x, y);
    vs[i++] = mid(x, y, y);
    vs[i++] = sin(x);
    vs[i++] = cos(x);
    vs[i++] = exp(x);
    vs[i++] = log(x);
    vs[i++] = pown(x, 2);
    vs[i++] = x * 2;
    vs[i++] = log2(x);
    vs[i++] = log10(x);
    vs[i++] = tan(x);
    vs[i++] = asin(x);
    vs[i++] = acos(x);
    vs[i++] = atan(x);
    vs[i++] = sinh(x);
    vs[i++] = cosh(x);
    vs[i++] = tanh(x);
    vs[i++] = asinh(x);
    vs[i++] = acosh(x);
    vs[i++] = atanh(x);
    vs[i++] = atan2(y, x);
    vs[i++] = atan2(y, 2.0);
    vs[i++] = atan2(2.0, x);
    vs[i++] = recip(x);
    vs[i++] = exp2(x);
    vs[i++] = expm1(x);
    vs[i++] = log1p(x);
    vs[i++] = hypot(x, y);
    vs[i++] = erf(x);
    vs[i++] = erfc(x);
    vs[i++] = midpoint(x, y);

    print(vs[out_i]);

    return vs[out_i];
}

__global__ void kernel(T *xs, T *ys, T *res, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res[i] = f(i, xs[i], ys[i]);
    }
}

int main()
{
    T xs[n], ys[n], res[n];

    // generate dummy data
    for (int i = 0; i < n; i++) {
        double v = i + 1;
        xs[i]    = { v, 1.0 };
        ys[i]    = { v, 0.0 };
    }

    T *d_xs, *d_ys, *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_ys, n * sizeof(*ys)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys, n * sizeof(*ys), cudaMemcpyHostToDevice));

    std::cout << "Results (GPU):\n";
    std::cout << "--------------\n";
    kernel<<<n, 1>>>(d_xs, d_ys, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    std::cout << '\n';
    std::cout << "Results (CPU):\n";
    std::cout << "--------------\n";
    for (int i = 0; auto el : res) {
        std::cout << std::format("[{:>2}] {:g}\n", i++, el);
    }

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
