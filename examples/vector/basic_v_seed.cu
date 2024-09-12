#include "../common.h"

#include <cuda_runtime.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <iostream>

using cu::tangents;

constexpr __device__ auto f(auto x, auto y, auto z, auto w)
{

    auto print = [](auto x) { printf("f {%g, [%g, %g]}\n", x.v, x.ds[0], x.ds[1]); };
    auto a     = x * y + z + w;
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

#if 0
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




// template<typename T, int N>
// __global__ void binary_op(T *in, T *out, int n_elems, int n_vars)
// {
//     auto print = [](auto x) { printf("[GPU] {%g, [%g, %g]}\n", x.v, x.ds[0], x.ds[1]); };
//     // TODO: this should maybe use the outer grid blocks?
//     //       and inside the kernels we can use the warps?
//     // int i = blockIdx.x * blockDim.x + threadIdx.x;
//     // int i = blockIdx.x * (blockDim.x * gridDim.x);
//     // int i = blockIdx.x;
//     // printf("i: %d\n", i);
//     // printf("blockIdx.x: %d\n", blockIdx.x);
//
//     // if (i < n) {
//
//     for (int i = blockIdx.x * blockDim.x; i < n_elems; i += blockDim.x * gridDim.x) {
//         // vtangent<T, N> res[n_vars];
//         extern __shared__ vtangent<T, N> res[]; // have to set kernel with n_vars
//
//         // seed
//         for (int j = 0; j < n_vars; j++) {
//             auto idx = i * n_vars + j;
//             printf("idx: %d\n", idx);
//             // res[idx].v = in[idx];
//             value(res[idx]) = in[idx];
//             printf("res[idx].v = %g\n", res[idx].v);
//
//             for (int k = 0; k < N; k++) {
//                 res[idx].ds[k] = k == j ? 1.0 : 0.0;
//             }
//         }
//
//         printf("[%d] Input a: {%g, [%g, %g]}\n", i, res[i * n_vars].v, res[i * n_vars].ds[0], res[i * n_vars].ds[1]);
//         printf("[%d] Input b: {%g, [%g, %g]}\n", i, res[i * n_vars + 1].v, res[i * n_vars + 1].ds[0], res[i * n_vars + 1].ds[1]);
//
//         auto res_v = f(res[i * n_vars], res[i * n_vars + 1], res[i * n_vars + 2], res[i * n_vars + 3]);
//
//         printf("Res_v\n");
//         print(res_v);
//
//         // harvest
//         int idx  = i * (n_vars + 1);
//         out[idx] = res_v.v;
//
//         for (int k = 0; k < N; k++) {
//             out[idx + k + 1] = res_v.ds[k];
//         }
//     }
// }
//

#elif 0

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

#elif 0

template<typename T, int N>
__global__ void binary_op(T *in, T *out, int n_elems, int n_vars)
{
    auto print = [](auto i, auto l, auto x) { printf("[%d][%d] {%g, [%g, %g]}\n", i, l, x.v, x.ds[0], x.ds[1]); };
    printf("grid dim.x = %d, block dim.x = %d\n", gridDim.x, blockDim.x);

    for (int i = blockIdx.x; i < n_elems; i += gridDim.x) {

        // vtangent<T, N> res[n_vars];
        extern __shared__ vtangent<T, N> xs[]; // have to set kernel with n_vars * sizeof(vtangent<T, N>)

        for (int l = 0; l < ceil(n_vars / N); l++) { // chunk wise vector tangent mode to eventually cover all basis vectors

            // seed
            for (int j = 0; j < n_vars; j++) {
                auto global_idx = i * n_vars + j;
                value(xs[j])    = in[global_idx];

                if (threadIdx.x == 0) {
                    printf("[%d][%d] global_idx: %d\n", i, l, global_idx);
                    printf("[%d][%d] res[%d].v = %g\n", i, l, j, xs[j].v);
                }

                for (int k = 0; k < N; k++) {
                    xs[j].ds[k] = l * N + k == j ? 1.0 : 0.0;
                }
            }

            vtangent<T, N> res_v = f(xs[0], xs[1], xs[2], xs[3]);

            if (threadIdx.x == 0) {
                printf("[%d][%d] Input a: {%g, [%g, %g]}\n", i, l, xs[0].v, xs[0].ds[0], xs[0].ds[1]);
                printf("[%d][%d] Input b: {%g, [%g, %g]}\n", i, l, xs[1].v, xs[1].ds[0], xs[1].ds[1]);
                printf("[%d][%d] Input c: {%g, [%g, %g]}\n", i, l, xs[2].v, xs[2].ds[0], xs[2].ds[1]);
                printf("[%d][%d] Input d: {%g, [%g, %g]}\n", i, l, xs[3].v, xs[3].ds[0], xs[3].ds[1]);
            }
            print(i, l, res_v);

            // harvest
            int idx = i * (n_vars + 1);
            int tid = threadIdx.x;

            // it might be beneficial to first coalesce the memory in shared memory
            // before copying it back to global memory
            if (tid + l * N < n_vars) {
                out[idx + tid + l * N + 1] = res_v.ds[tid];
            }

            if (tid == 0) {
                out[idx] = res_v.v;
            }

            // for (int k = 0; k < N and k + l * N < n_vars; k++) {
            //     out[idx + k + l * N + 1] = res_v.ds[k];
            // }
        }
    }
}

#else
// N is the number or simultaneous tangent computations
// n_elems the number of elements
// n_vars the number of variables

template<typename T, int N>
__global__ void binary_op(T *in, T *out, int n_elems, int n_vars)
{
    auto print = [](auto i, auto l, auto x) { printf("[%d][%d] {%g, [%g, %g]}\n", i, l, x.v, x.ds[0], x.ds[1]); };
    // printf("grid dim.x = %d, block dim.x = %d\n", gridDim.x, blockDim.x);

    constexpr int n_elems_per_block = 2; // Process 2 elements per block

    for (int q = blockIdx.x * n_elems_per_block; q < n_elems; q += gridDim.x * n_elems_per_block) {
        for (int i = q; (i < q + n_elems_per_block) and (i < n_elems); i++) {

            // vtangent<T, N> res[n_vars];
            extern __shared__ tangents<T, N> xs[]; // have to set kernel with n_vars * sizeof(vtangent<T, N>)

            for (int l = 0; l < ceil(n_vars / N); l++) { // chunk wise vector tangent mode to eventually cover all basis vectors

                // seed
                for (int j = 0; j < n_vars; j++) {
                    auto global_idx = i * n_vars + j;
                    value(xs[j])    = in[global_idx];

                    if (threadIdx.x == 0) {
                        printf("[%d][%d] global_idx: %d\n", i, l, global_idx);
                        printf("[%d][%d] res[%d].v = %g\n", i, l, j, xs[j].v);
                    }

                    for (int k = 0; k < N; k++) {
                        xs[j].ds[k] = l * N + k == j ? 1.0 : 0.0;
                    }
                }

                tangents<T, N> res_v = f(xs[0], xs[1], xs[2], xs[3]);

                if (threadIdx.x == 0) {
                    printf("[%d][%d] Input a: {%g, [%g, %g]}\n", i, l, xs[0].v, xs[0].ds[0], xs[0].ds[1]);
                    printf("[%d][%d] Input b: {%g, [%g, %g]}\n", i, l, xs[1].v, xs[1].ds[0], xs[1].ds[1]);
                    printf("[%d][%d] Input c: {%g, [%g, %g]}\n", i, l, xs[2].v, xs[2].ds[0], xs[2].ds[1]);
                    printf("[%d][%d] Input d: {%g, [%g, %g]}\n", i, l, xs[3].v, xs[3].ds[0], xs[3].ds[1]);
                }
                print(i, l, res_v);

                // harvest
                int idx = i * (n_vars + 1);
                int tid = threadIdx.x;

                // it might be beneficial to first coalesce the memory in shared memory
                // before copying it back to global memory

                // first case to only copy the results from the seeded input variables to output
                // second case to ensure that we do not go over the size of the output vector
                if (tid < N and tid + l * N < n_vars) {
                    out[idx + tid + l * N + 1] = res_v.ds[tid];
                }

                if (tid == 0) {
                    out[idx] = res_v.v;
                }

                // for (int k = 0; k < N and k + l * N < n_vars; k++) {
                //     out[idx + k + l * N + 1] = res_v.ds[k];
                // }
            }
        }
    }
}

#endif

int main()
{
    constexpr int n_elems = 10;
    constexpr int n_vars  = 4;

    // constexpr int n = n_elems;

    using T = double;
    T xs[n_elems * n_vars], /* ys[n], */ res[n_elems * (1 + n_vars)];

    memset(xs, 0, n_elems * n_vars * sizeof(*xs));

    // generate dummy data
    for (int i = 0; i < n_elems * n_vars; i += n_vars) {
        double v  = i + 2;
        xs[i]     = v;
        xs[i + 1] = v + 1;
        xs[i + 2] = v + 2;
        xs[i + 3] = v + 3;
    }

    for (int i = 0; i < n_elems * n_vars; i++) {
        printf("x is : %g\n", xs[i]);
    }

    T *d_xs, /* *d_ys, */ *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n_elems * n_vars * sizeof(*xs)));
    // CUDA_CHECK(cudaMalloc(&d_ys, n_elems * sizeof(*ys)));
    CUDA_CHECK(cudaMalloc(&d_res, (n_elems * (1 + n_vars)) * sizeof(*res)));

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n_elems * n_vars * sizeof(*xs), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_ys, ys, n_elems * sizeof(*ys), cudaMemcpyHostToDevice));

    // seed<<<n_elems, 1>>>(d_xs, d_res, n_elems, n_vars);
    binary_op<T, 2><<<2, 8, n_vars * sizeof(tangents<T, 2>)>>>(d_xs, d_res, n_elems, n_vars);
    // harvest

    CUDA_CHECK(cudaMemcpy(res, d_res, (n_elems * (1 + n_vars)) * sizeof(*res), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (auto el : res) {
        std::cout << el << std::endl;
    }

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}

// int main()
// {
//     constexpr int n_elems = 2;
//     constexpr int n_vars  = 4;
//
//     // constexpr int n = n_elems;
//
//     using T = double;
//     T xs[n_elems * n_vars], /* ys[n], */ res[n_elems * (1 + n_vars)];
//
//     memset(xs, 0, n_elems * n_vars * sizeof(*xs));
//
//     // generate dummy data
//     for (int i = 0; i < n_elems * n_vars; i += n_vars) {
//         double v  = i + 2;
//         xs[i]     = v;
//         xs[i + 1] = v + 1;
//     }
//
//     for (int i = 0; i < n_elems * n_vars; i++) {
//         printf("x is : %g\n", xs[i]);
//     }
//
//     // for (int i = 0; i < n_elems; i++) {
//     //     double v = i + 3;
//     //     xs[i + n_elems]    = v;
//     // }
//
//     T *d_xs, /* *d_ys, */ *d_res;
//     CUDA_CHECK(cudaMalloc(&d_xs, n_elems * n_vars * sizeof(*xs)));
//     // CUDA_CHECK(cudaMalloc(&d_ys, n_elems * sizeof(*ys)));
//     CUDA_CHECK(cudaMalloc(&d_res, (n_elems * (1 + n_vars)) * sizeof(*res)));
//
//     CUDA_CHECK(cudaMemcpy(d_xs, xs, n_elems * n_vars * sizeof(*xs), cudaMemcpyHostToDevice));
//     // CUDA_CHECK(cudaMemcpy(d_ys, ys, n_elems * sizeof(*ys), cudaMemcpyHostToDevice));
//
//     // seed<<<n_elems, 1>>>(d_xs, d_res, n_elems, n_vars);
//     binary_op<double, n_vars><<<16, 8>>>(d_xs, d_res, n_elems);
//     // harvest
//
//     CUDA_CHECK(cudaMemcpy(res, d_res, (n_elems * (1 + n_vars)) * sizeof(*res), cudaMemcpyDeviceToHost));
//     for (auto el : res) {
//         std::cout << el << std::endl;
//     }
//     CUDA_CHECK(cudaDeviceSynchronize());
//
//     CUDA_CHECK(cudaFree(d_xs));
//     CUDA_CHECK(cudaFree(d_res));
//
//     return 0;
// }
