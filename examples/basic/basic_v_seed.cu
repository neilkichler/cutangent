#include "../common.h"

#include <cuda_runtime.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <iostream>

using cu::vtangent;

constexpr auto f(auto x, auto y, auto z)
{
    auto print = [](auto x) { printf("{%g, [%g, %g]}\n", x.v, x.ds[0], x.ds[1]); };
    // auto a     = x[0] + x[1];
    auto a = x * y + z;
    print(a);
    return a;
}

constexpr auto g(auto x)
{
    return -x;
}

// int n_vars = N; // Not the case if N < n_vars
// for (int j = 0; j < n_vars; j++) {
//     xs[i].ds[j] = j == 0 ? 1.0 : 0.0;
//     ys[i].ds[j] = j == 1 ? 1.0 : 0.0;
//
// res[i] = f(xs[i], ys[i]);
// }

// template<typename T>
// __global__ void unary_op(T *xs, T *res, int n)
// {
//     auto print = [](auto x) { printf("[GPU] {%g, [%g]}\n", x.v, x.ds[0]); };
//     int i      = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) {
//         vtangent<T, 1> xs_v;
//
//         xs_v.v = xs[i];
//
//         xs_v.ds[0] = 1.0;
//
//         auto res_v = g(xs_v);
//
//         print(res_v);
//
//         res[i * 2 + 0] = res_v.v;
//         res[i * 2 + 1] = res_v.ds[0];
//     }
// }
//

#if 1
template<typename T, int N>
__global__ void binary_op(T *in, T *out, int n)
{
    auto print = [](auto x) { printf("[GPU] {%g, [%g, %g]}\n", x.v, x.ds[0], x.ds[1]); };
    int i      = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        vtangent<T, N> res[N];

        // seed
        for (int j = 0; j < N; j++) {
            auto idx = i * N + j;
            printf("idx: %d\n", idx);
            res[idx].v = in[idx];
            printf("res[idx].v = %g\n", res[idx].v);

            for (int k = 0; k < N; k++) {
                res[idx].ds[k] = k == j ? 1.0 : 0.0;
            }
        }

        printf("[%d] Input a: {%g, [%g, %g]}\n", i, res[i * N].v, res[i * N].ds[0], res[i * N].ds[1]);
        printf("[%d] Input b: {%g, [%g, %g]}\n", i, res[i * N + 1].v, res[i * N + 1].ds[0], res[i * N + 1].ds[1]);

        auto res_v = f(res[i * N], res[i * N + 1], res[i * N + 2]);

        printf("Res_v\n");
        print(res_v);

        // harvest
        int idx  = i * (N + 1);
        out[idx] = res_v.v;

        for (int j = 0; j < N; j++) {
            out[idx + j + 1] = res_v.ds[j];
        }
    }
}

#else

// In only contains the values, the seeding is always done with
// the cartesian basis coordinates.
template<typename T, int N>
__global__ void seed(T *in, vtangent<T, N> *out, int n_elems, int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {

        for (int j = 0; j < n_vars; j++) {
            auto idx   = i * n_vars + j;
            out[idx].v = in[idx];

            for (int k = 0; k < n_vars; k++) {
                out[idx].ds[k] = k == j ? 1.0 : 0.0;
            }
        }

        // here we should do the compute

        // then directly the harvest
    }
}

template<typename T, int N>
__global__ void harvest(vtangent<T, N> *in, T *out, int n_elems, int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {

        int idx  = i * (n_vars + 1);
        out[idx] = in[i * n_vars].v;

        for (int j = 0; j < n_vars; j++) {
            out[idx + j] = in[i * n_vars].ds[j];
        }
    }
}

template<typename T, int N>
__global__ void binary_op(T *in, T *out, int n)
{
    auto print = [](auto x) { printf("[GPU] {%g, [%g, %g]}\n", x.v, x.ds[0], x.ds[1]); };
    int i      = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        vtangent<T, N> res[N];

        // seed
        // for (int j = 0; j < N; j++) {
        //     auto idx = i * N + j;
        //     printf("idx: %d\n", idx);
        //     res[idx].v = in[idx];
        //     printf("res[idx].v = %g\n", res[idx].v);
        //
        //     for (int k = 0; k < N; k++) {
        //         res[idx].ds[k] = k == j ? 1.0 : 0.0;
        //     }
        // }

        printf("[%d] Input a: {%g, [%g, %g]}\n", i, res[i * N].v, res[i * N].ds[0], res[i * N].ds[1]);
        printf("[%d] Input b: {%g, [%g, %g]}\n", i, res[i * N + 1].v, res[i * N + 1].ds[0], res[i * N + 1].ds[1]);

        auto res_v = f(res[i * N], res[i * N + 1], res[i * N + 2]);

        printf("Res_v\n");
        print(res_v);

        // // harvest
        // int idx  = i * (N + 1);
        // out[idx] = res_v.v;
        //
        // for (int j = 0; j < N; j++) {
        //     out[idx + j + 1] = res_v.ds[j];
        // }
    }
}
#endif

int main()
{
    constexpr int n_elems = 5;
    constexpr int n_vars  = 3;

    // constexpr int n = n_elems;

    using T = double;
    T xs[n_elems * n_vars], /* ys[n], */ res[n_elems * (1 + n_vars)];

    memset(xs, 0, n_elems * n_vars * sizeof(*xs));

    // generate dummy data
    for (int i = 0; i < n_elems * n_vars; i += n_vars) {
        double v  = i + 2;
        xs[i]     = v;
        xs[i + 1] = v + 1;
    }

    for (int i = 0; i < n_elems * n_vars; i++) {
        printf("x is : %g\n", xs[i]);
    }

    // for (int i = 0; i < n_elems; i++) {
    //     double v = i + 3;
    //     xs[i + n_elems]    = v;
    // }

    T *d_xs, /* *d_ys, */ *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n_elems * n_vars * sizeof(*xs)));
    // CUDA_CHECK(cudaMalloc(&d_ys, n_elems * sizeof(*ys)));
    CUDA_CHECK(cudaMalloc(&d_res, (n_elems * (1 + n_vars)) * sizeof(*res)));

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n_elems * n_vars * sizeof(*xs), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_ys, ys, n_elems * sizeof(*ys), cudaMemcpyHostToDevice));

    // seed<<<n_elems, 1>>>(d_xs, d_res, n_elems, n_vars);
    binary_op<double, n_vars><<<n_elems, 1>>>(d_xs, d_res, n_elems);
    // harvest

    CUDA_CHECK(cudaMemcpy(res, d_res, (n_elems * (1 + n_vars)) * sizeof(*res), cudaMemcpyDeviceToHost));
    for (auto el : res) {
        std::cout << el << std::endl;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
