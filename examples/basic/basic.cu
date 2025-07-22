#include "../common.h"

#include <cuda_runtime.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <cstdio>
#include <vector>

using cu::tangent;

constexpr auto f(auto x, auto y)
{
    auto print = [](auto x) { printf("{%g, %g}\n", x.v, x.d); };

    int i = 0;
    std::vector<tangent<double>> vs(35);

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

    for (auto v : vs) {
        print(v);
    }

    return vs[0];
}

__global__ void kernel(tangent<double> *xs, tangent<double> *ys,
                       tangent<double> *res, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res[i] = f(xs[i], ys[i]);
    }
}

int main()
{
    constexpr int n = 16;
    using T         = tangent<double>;
    T xs[n], ys[n], res[n];

    // generate dummy data
    for (int i = 0; i < n; i++) {
        double v = i;
        xs[i]    = { v, 1.0 };
        ys[i]    = { v, 0.0 };
    }

    // for (auto el : xs) {
    //     std::cout << el << std::endl;
    // }

    T *d_xs, *d_ys, *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_ys, n * sizeof(*ys)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys, n * sizeof(*ys), cudaMemcpyHostToDevice));

    kernel<<<n, 1>>>(d_xs, d_ys, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    // for (auto el : res) {
    //     std::cout << el << std::endl;
    // }

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
