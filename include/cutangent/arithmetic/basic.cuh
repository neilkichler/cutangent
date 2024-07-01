#ifndef CUTANGENT_ARITHMETIC_BASIC_CUH
#define CUTANGENT_ARITHMETIC_BASIC_CUH

#include <cutangent/tangent.h>

#include <algorithm>
// #include <cmath>
// #include <numbers>

namespace cu
{

#define cuda_fn inline constexpr __device__

template<typename T>
cuda_fn cu::tangent<T> operator+(cu::tangent<T> a, cu::tangent<T> b)
{
    return { .v = a.v + b.v, .d = a.d + b.d };
}

template<typename T>
cuda_fn cu::tangent<T> operator*(cu::tangent<T> a, cu::tangent<T> b)
{
    return { .v = a.v * b.v, .d = a.v * b.d + a.d * b.v };
}

template<typename T>
cuda_fn cu::tangent<T> max(cu::tangent<T> a, cu::tangent<T> b)
{
    using std::max;

    return { .v = max(a.v, b.v),
             .d = a.v >= b.v ? a.d : b.d }; // '>=' instead of '>' due to subgradient
}

template<typename T>
cuda_fn cu::tangent<T> min(cu::tangent<T> a, cu::tangent<T> b)
{
    using std::min;

    return { .v = min(a.v, b.v),
             .d = a.v <= b.v ? a.d : b.d }; // '<=' instead of '<' due to subgradient
}

template<typename T>
cuda_fn cu::tangent<T> mid(cu::tangent<T> v, cu::tangent<T> lb, cu::tangent<T> ub)
{
    return { .v = mid(v.v, lb.v, ub.v), .d = lb.d * (v.v < lb.v) + ub.d * (v.v > ub.v) };
}

} // namespace cu

#endif // CUTANGENT_ARITHMETIC_BASIC_CUH
