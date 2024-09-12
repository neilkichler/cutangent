#include "../common.h"

#include <cuda_runtime.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <iostream>

using cu::tangents;

constexpr auto f(auto x, auto y)
{
    auto print = [](auto x) { printf("[GPU] {%g, [%g, %g]}\n", x.v, x.ds[0], x.ds[1]); };

    auto a = x * y;
    print(a);
    return a;
}

constexpr auto g(auto x)
{
    return -x;
}

template<typename T>
__global__ void unary_op(T *xs, T *res, int n)
{
    auto print = [](auto x) { printf("[GPU] {%g, [%g]}\n", x.v, x.ds[0]); };
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        tangents<T, 1> xs_v;

        xs_v.v = xs[i];

        xs_v.ds[0] = 1.0;

        auto res_v = g(xs_v);

        print(res_v);

        res[i * 2 + 0] = res_v.v;
        res[i * 2 + 1] = res_v.ds[0];
    }
}

template<typename T>
__global__ void binary_op(T *xs, T *ys, T *res, int n)
{
    auto print = [](auto x) { printf("[GPU] {%g, [%g, %g]}\n", x.v, x.ds[0], x.ds[1]); };
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        tangents<T, 2> xs_v;
        tangents<T, 2> ys_v;

        xs_v.v = xs[i];
        ys_v.v = ys[i];

        xs_v.ds[0] = 1.0;
        xs_v.ds[1] = 0.0;
        ys_v.ds[0] = 0.0;
        ys_v.ds[1] = 1.0;

        auto res_v = f(xs_v, ys_v);

        print(res_v);

        res[i * 3 + 0] = res_v.v;
        res[i * 3 + 1] = res_v.ds[0];
        res[i * 3 + 2] = res_v.ds[1];
    }
}

int main()
{
    constexpr int n_elems = 10;
    constexpr int n_vars  = 2;

    constexpr int n = n_elems;

    using T = double;
    T xs[n], ys[n], res[n_elems * (1 + n_vars)];

    // generate dummy data
    for (int i = 0; i < n_elems; i++) {
        double v = i + 2;
        xs[i]    = v;
    }

    for (int i = 0; i < n_elems; i++) {
        double v = i + 3;
        ys[i]    = v;
    }

    T *d_xs, *d_ys, *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n_elems * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_ys, n_elems * sizeof(*ys)));
    CUDA_CHECK(cudaMalloc(&d_res, (n_elems * (1 + n_vars)) * sizeof(*res)));

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n_elems * sizeof(*xs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys, n_elems * sizeof(*ys), cudaMemcpyHostToDevice));

    binary_op<<<n_elems, 1>>>(d_xs, d_ys, d_res, n_elems);
    CUDA_CHECK(cudaMemcpy(res, d_res, (n_elems * (1 + n_vars)) * sizeof(*res), cudaMemcpyDeviceToHost));
    for (auto el : res) {
        std::cout << el << std::endl;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    unary_op<<<n_elems, 1>>>(d_xs, d_res, n_elems);
    CUDA_CHECK(cudaMemcpy(res, d_res, (n_elems * (1 + n_vars - 1)) * sizeof(*res), cudaMemcpyDeviceToHost));
    for (auto el : res) {
        std::cout << el << std::endl;
    }

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
