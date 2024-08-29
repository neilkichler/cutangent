#include "../common.h"

#include <cuda_runtime.h>

#include <cuinterval/cuinterval.h>
#include <cuinterval/format.h>

#include <cutangent/cutangent.cuh>

#include <iostream>

using cu::tangent;

using I = cu::interval<double>;

using T = tangent<I>;

constexpr auto f(auto x)
{
    using std::pow;

    return 3.0 * pow(x, 3) + pow(x, 2) - 5.0 * x - 1.0;
}

__global__ void centered_form(T *xs, I *res, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        auto r = f(xs[i]);

        auto L = derivative(r);
        auto X = value(xs[i]);
        auto c = mid(value(xs[i]));

        auto centered_form = f(c) + L * (X - c);

        res[i] = centered_form;
    }
}

int main()
{
    constexpr int n = 1;
    T xs[n];
    I res[n];

    value(xs[0])      = { -1.0, 1.0 };
    derivative(xs[0]) = { 1.0, 1.0 };

    T *d_xs;
    I *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));

    centered_form<<<n, 1>>>(d_xs, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    auto r = res[0];
    std::cout << "f(x) = x^3 + x^2 - 5x - 1" << std::endl;
    std::cout << "X = " << value(xs[0]) << std::endl;
    std::cout << "centered form for f at X: " << r << std::endl;

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
