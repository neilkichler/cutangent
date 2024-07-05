#ifndef CUTANGENT_ARITHMETIC_BASIC_CUH
#define CUTANGENT_ARITHMETIC_BASIC_CUH

#include <cutangent/tangent.h>

#include <algorithm>
#include <cmath>

namespace cu
{

#define fn inline constexpr __device__

template<typename T>
fn tangent<T> operator-(tangent<T> x)
{
    return { -x.v, -x.d };
}

template<typename T>
fn tangent<T> operator+(tangent<T> a, tangent<T> b)
{
    return { .v = a.v + b.v, .d = a.d + b.d };
}

template<typename T>
fn tangent<T> operator-(tangent<T> a, tangent<T> b)
{
    return { .v = a.v - b.v, .d = a.d - b.d };
}

template<typename T>
fn tangent<T> operator*(tangent<T> a, tangent<T> b)
{
    return { .v = a.v * b.v, .d = a.v * b.d + a.d * b.v };
}

template<typename T>
fn tangent<T> operator/(tangent<T> a, tangent<T> b)
{
    return { .v = a.v / b.v, .d = (a.d * b.v - a.v * b.d) / (b.v * b.v) };
}

template<typename T>
fn tangent<T> max(tangent<T> a, tangent<T> b)
{
    using std::max;

    return { .v = max(a.v, b.v),
             .d = a.v >= b.v ? a.d : b.d }; // '>=' instead of '>' due to subgradient
}

template<typename T>
fn tangent<T> min(tangent<T> a, tangent<T> b)
{
    using std::min;

    return { .v = min(a.v, b.v),
             .d = a.v <= b.v ? a.d : b.d }; // '<=' instead of '<' due to subgradient
}

template<typename T>
fn tangent<T> clamp(tangent<T> v, tangent<T> lb, tangent<T> ub)
{
    using std::clamp;

    return { .v = clamp(v.v, lb.v, ub.v), .d = lb.d * (v.v < lb.v) + ub.d * (v.v > ub.v) };
}

template<typename T>
fn tangent<T> mid(tangent<T> v, tangent<T> lb, tangent<T> ub)
{
    return clamp(v, lb, ub);
}

template<typename T>
fn tangent<T> sin(tangent<T> x)
{
    using std::cos;
    using std::sin;

    return { .v = sin(x.v), .d = cos(x.v) * x.d };
}

template<typename T>
fn tangent<T> cos(tangent<T> x)
{
    using std::cos;
    using std::sin;

    return { .v = cos(x.v), .d = -sin(x.v) * x.d };
}

template<typename T>
fn tangent<T> exp(tangent<T> x)
{
    using std::exp;

    return { .v = exp(x.v), .d = exp(x.v) * x.d };
}

template<typename T>
fn tangent<T> log(tangent<T> x)
{
    using std::log;

    return { .v = log(x.v), .d = x.d / x.v };
}

template<typename T>
fn tangent<T> sqr(tangent<T> x)
{
    return { .v = sqr(x.v), .d = 2.0 * x.v * x.d };
}

template<typename T>
fn tangent<T> pown(tangent<T> x, auto n)
{
    using std::pow;

    return { .v = pow(x.v, n), .d = n * pow(x.v, n - 1) * x.d };
}

#undef fn

} // namespace cu

#endif // CUTANGENT_ARITHMETIC_BASIC_CUH
