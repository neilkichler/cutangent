#include "../common.h"

#include <cuda_runtime.h>

#include <cuinterval/arithmetic/intrinsic.cuh>

#include <cumccormick/cumccormick.cuh>
#include <cumccormick/format.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <iostream>

using cu::tangent;

template<typename T>
using mc = cu::mccormick<T>;

constexpr auto f(auto x, auto y)
{
    // Currently supported functions:

    // auto a = neg(x);
    // auto a = add(x, y);
    // auto a = sub(x, y);
    // auto a = mul(x, y);
    // auto a = div(x, y);
    // auto a = x + y;
    // auto a = x - y;
    // auto a = x / y;
    // auto a = x * y;
    // auto a = sqr(x);
    // auto a = sqrt(x);
    // auto a = abs(x);
    // auto a = exp(x);
    // auto a = log(x);
    // auto a = pown(x, 3);
    // auto a = pown(x, 4);
    // auto a = pow(x, 4);
    // auto a = pow(x, y); // not yet supported by McCormick
    // auto a = max(x, y);
    // auto a = min(x, y);
    auto a = hull(x, y);
    return a;
}

__global__ void kernel(mc<tangent<double>> *xs, mc<tangent<double>> *ys, mc<tangent<double>> *res, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res[i] = f(xs[i], ys[i]);
    }
}

int main()
{
    constexpr int n = 1;
    using T         = mc<tangent<double>>;
    T xs[n], ys[n], zs[n], res[n];

    // generate dummy data

    xs[0] = { .cv  = { 2.0, 1.0 },
              .cc  = { 2.0, 1.0 },
              .box = { .lb = { 0.5, 0.0 },
                       .ub = { 3.0, 0.0 } } };

    ys[0] = { .cv  = { 3.0, 1.0 },
              .cc  = { 3.0, 1.0 },
              .box = { .lb = { 2.0, 0.0 },
                       .ub = { 5.0, 0.0 } } };

    zs[0] = { .cv  = { -3.0, 1.0 },
              .cc  = { -3.0, 1.0 },
              .box = { .lb = { -12.0, 0.0 },
                       .ub = { 5.0, 0.0 } } };

    std::cout << xs[0] << std::endl;
    std::cout << ys[0] << std::endl;

    T *d_xs, *d_ys, *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_ys, n * sizeof(*ys)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys, n * sizeof(*ys), cudaMemcpyHostToDevice));

    kernel<<<n, 1>>>(d_xs, d_ys, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    auto r = res[0];
    std::cout << r << std::endl;

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
