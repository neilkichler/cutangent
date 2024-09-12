#include "../common.h"

#include <cuda_runtime.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <iostream>
#include <vector>

using cu::tangents;

constexpr auto f(auto x, auto y, auto z)
{
    auto print = [](auto x) { printf("{%g, [%g, %g]}\n", x.v, x.ds[0], x.ds[1]); };
    auto a     = x * y + z;
    print(a);
    return a;
}

constexpr auto g(auto x)
{
    return -x;
}

template<typename T, int N>
__global__ void binary_op(T *in, T *out, int n, cudaGraphExec_t cgraph)
{
    auto print = [](auto x) { printf("[GPU] {%g, [%g, %g]}\n", x.v, x.ds[0], x.ds[1]); };
    int i      = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        tangents<T, N> res[N];

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

        cudaGraphLaunch(cgraph, 0);

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

// template<typename T>
// cudaGraph_t construct_graph()
// {
//
// }

int main()
{
    constexpr int n_elems = 5;
    constexpr int n_vars  = 3;

    constexpr int n_blocks  = n_elems;
    constexpr int n_threads = 1;

    // constexpr int n = n_elems;

    using T = double;
    T xs[n_elems * n_vars], res[n_elems * (1 + n_vars)];

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

    T *d_xs, *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n_elems * n_vars * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, (n_elems * (1 + n_vars)) * sizeof(*res)));
    CUDA_CHECK(cudaMemcpy(d_xs, xs, n_elems * n_vars * sizeof(*xs), cudaMemcpyHostToDevice));

    cudaGraph_t graph;
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    using node = cudaGraphNode_t;

    std::vector<node> dependencies;
    node node_kernel;

    cudaKernelNodeParams kernel_params {
        .func           = nullptr,
        .gridDim        = dim3(n_blocks, 1, 1),
        .blockDim       = dim3(n_threads, 1, 1),
        .sharedMemBytes = 0,
        .kernelParams   = nullptr,
        .extra          = nullptr
    };

    // manually create a * b + c from individual device kernels
    // auto v0 = d_xs[0];
    // auto v1 = d_xs[1];
    // auto v2 = d_xs[2];
    // {
    //     void *params[4]            = { &v0, &v1, &d_res, &n_k };
    //     kernel_params.func         = (void *)mul<tangents<T, n_vars>>;
    //     kernel_params.kernelParams = params;
    //
    //     CUDA_CHECK(cudaGraphAddKernelNode(&node_kernel, graph, dependencies.data(), dependencies.size(), &kernel_params));
    //     dependencies.clear();
    //     dependencies.push_back(node_kernel);
    // }

    cudaGraphExec_t xgraph;
    CUDA_CHECK(cudaGraphInstantiate(&xgraph, graph, nullptr, nullptr, 0));

    binary_op<double, n_vars><<<n_blocks, n_threads>>>(d_xs, d_res, n_elems, xgraph);

    CUDA_CHECK(cudaMemcpy(res, d_res, (n_elems * (1 + n_vars)) * sizeof(*res), cudaMemcpyDeviceToHost));
    for (auto el : res) {
        std::cout << el << std::endl;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
