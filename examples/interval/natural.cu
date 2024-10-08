#include "../common.h"

#include <cuda_runtime.h>

#include <cuinterval/cuinterval.h>
#include <cuinterval/format.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <iostream>

using cu::tangent;

template<typename T>
using I = cu::interval<T>;

using T = tangent<I<double>>;

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
    auto a = x * y;
    // auto a = sqr(x);
    // auto a = sqrt(x);
    // auto a = abs(x);
    // auto a = exp(x);
    // auto a = log(x);
    // auto a = recip(x);
    // auto a = cos(x);
    // auto a = pow(x, 3);
    // auto a = pown(x, 4.0);
    // auto a = pow(x, 4.0);
    // auto a = pow(x, y);
    // auto a = atan2(y, x);
    // auto a = atan2(y, 2.0);
    // auto a = atan2(2.0, x);
    // auto a = max(x, y);
    // auto a = min(x, y);
    // auto a = hull(x, y);
    return a;
}

__global__ void kernel(T *xs, T *ys, T *res, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res[i] = f(xs[i], ys[i]);
    }
}

int main()
{
    constexpr int n = 1;
    T xs[n], ys[n], res[n];

    // generate dummy data
    value(xs[0])      = { 0.5, 3.0 };
    derivative(xs[0]) = { 1.0, 1.0 };
    value(ys[0])      = { 2.0, 5.0 };
    derivative(ys[0]) = { 0.0, 0.0 };

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
