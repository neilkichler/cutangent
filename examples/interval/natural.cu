#include "../common.h"

#include <cuda_runtime.h>

#include <cuinterval/cuinterval.h>
#include <cuinterval/format.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <iostream>

constexpr auto f(auto x, auto y)
{
    using T = decltype(x);

    auto a = x;

    // Currently supported functions:
    a = -x;
    a = x + y;
    a = x - y;
    a = x / y;
    a = x * y;
    a = sqr(x);
    a = sqrt(x);
    a = cbrt(x);
    // a = abs(x);
    a = exp(x);
    a = log(x);
    a = log2(x);
    a = log10(x);
    a = pown(x, 4);
    a = pow(x, 3);
    a = pow(x, 4.0f);
    a = pow(x, T(4.0));
    a = pow(x, y);
    a = recip(x);
    a = sin(x);
    a = cos(x);
    a = tan(x);
    a = asin(x);
    a = atan(x);
    a = atan2(y, x);
    a = atan2(y, T(2.0));
    a = atan2(T(2.0), x);
    a = sinh(x);
    a = cosh(x);
    a = asinh(x);
    a = acosh(x);
    a = atanh(x);
    a = erf(x);
    a = erfc(x);
    // a = max(x, y);
    // a = min(x, y);
    a = ceil(x);
    a = floor(x);
    a = trunc(x);
    a = round(x);
    a = nearbyint(x);
    a = rint(x);

    // int dummy;
    // a = remquo(x, y, &dummy);
    // a = hull(x, y);

    if (isinf(x) || isfinite(x)) {
        a = 2 * a;
    }

    return a;
}

template<typename T>
__global__ void kernel(T *xs, T *ys, T *res, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res[i] = f(xs[i], ys[i]);
    }
}

template<typename T>
void test_all_ops()
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
}

int main()
{
    using cu::interval, cu::tangent;

    test_all_ops<tangent<interval<float>>>();
    test_all_ops<tangent<interval<double>>>();

    return 0;
}
