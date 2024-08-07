#include "../common.h"

#include <cuda_runtime.h>

#include <cutangent/cutangent.cuh>
#include <cutangent/format.h>

#include <iostream>

using cu::tangent;

constexpr auto f(auto x, auto y)
{
    auto print = [](auto x) { printf("{%g, %g}\n", x.v, x.d); };

    auto a  = x + y;
    auto b  = x - y;
    auto c  = x * y;
    auto d  = x / y;
    auto e  = max(x, y);
    auto f  = min(x, y);
    auto g  = mid(x, y, y);
    auto h  = sin(x);
    auto i  = cos(x);
    auto j  = exp(x);
    auto k  = log(x);
    auto l  = pown(x, 2);
    auto m  = x * 2;
    auto n  = log2(x);
    auto o  = log10(x);
    auto p  = tan(x);
    auto q  = asin(x);
    auto r  = acos(x);
    auto s  = atan(x);
    auto t  = sinh(x);
    auto u  = cosh(x);
    auto v  = tanh(x);
    auto w  = asinh(x);
    auto aa = acosh(x);
    auto bb = atanh(x);
    auto cc = atan2(y, x);
    auto dd = atan2(y, 2.0);
    auto ee = atan2(2.0, x);

    print(a);
    print(b);
    print(c);
    print(d);
    print(e);
    print(f);
    print(g);
    print(h);
    print(i);
    print(j);
    print(k);
    print(l);
    print(m);
    print(n);
    print(o);
    print(p);
    print(q);
    print(r);
    print(s);
    print(t);
    print(u);
    print(v);
    print(w);
    print(aa);
    print(bb);
    print(cc);
    print(dd);
    print(ee);
    return a;
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
