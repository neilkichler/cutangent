#ifndef CUTANGENT_ARITHMETIC_BASIC_CUH
#define CUTANGENT_ARITHMETIC_BASIC_CUH

#include <cutangent/tangent.h>

#include <algorithm>
#include <cmath>
#include <concepts>

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
    return { a.v + b.v, a.d + b.d };
}

template<typename T>
fn tangent<T> operator-(tangent<T> a, tangent<T> b)
{
    return { a.v - b.v, a.d - b.d };
}

template<typename T>
fn tangent<T> operator*(tangent<T> a, tangent<T> b)
{
    return { a.v * b.v, a.v * b.d + a.d * b.v };
}

template<typename T>
fn tangent<T> operator*(tangent<T> a, T b)
{
    return { a.v * b, a.d * b };
}

template<typename T>
fn tangent<T> operator*(T a, tangent<T> b)
{
    return { a * b.v, a * b.d };
}

template<typename T>
fn tangent<T> operator/(tangent<T> a, tangent<T> b)
{
    return { a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v) };
}

template<typename T>
fn tangent<T> max(tangent<T> a, tangent<T> b)
{
    using std::max;

    return { max(a.v, b.v),
             a.v >= b.v ? a.d : b.d }; // '>=' instead of '>' due to subgradient
}

template<typename T>
fn tangent<T> min(tangent<T> a, tangent<T> b)
{
    using std::min;

    return { min(a.v, b.v),
             a.v <= b.v ? a.d : b.d }; // '<=' instead of '<' due to subgradient
}

template<typename T>
fn tangent<T> abs(tangent<T> x)
{
    using std::abs;
    using std::copysign;
    // NOTE: not differentiable at x = 0.
    return { abs(x.v), copysign(1.0, x.v) * x.d };
}

template<typename T>
fn tangent<T> clamp(tangent<T> v, tangent<T> lb, tangent<T> ub)
{
    using std::clamp;

    return { clamp(v.v, lb.v, ub.v), lb.d * (v.v < lb.v) + ub.d * (v.v > ub.v) };
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

    return { sin(x.v), cos(x.v) * x.d };
}

template<typename T>
fn tangent<T> cos(tangent<T> x)
{
    using std::cos;
    using std::sin;

    return { cos(x.v), -sin(x.v) * x.d };
}

template<typename T>
fn tangent<T> exp(tangent<T> x)
{
    using std::exp;

    return { exp(x.v), exp(x.v) * x.d };
}

template<typename T>
fn tangent<T> log(tangent<T> x)
{
    using std::log;

    // NOTE: We currently do not treat the case where x.v == 0, x.d != 0 to map to -+inf.
    return { log(x.v), x.d / x.v };
}

template<typename T>
fn tangent<T> sqr(tangent<T> x)
{
    return { sqr(x.v), 2.0 * x.v * x.d };
}

template<typename T>
fn tangent<T> sqrt(tangent<T> x)
{
    using std::sqrt;
    // NOTE: We currently do not treat the case where x.v == 0, x.d > 0 to map to +inf.
    return { sqrt(x.v), x.d / (2.0 * sqrt(x.v)) };
}

template<typename T>
fn tangent<T> pown(tangent<T> x, auto n)
{
    using std::pow;

    return { pow(x.v, n), n * pow(x.v, n - 1) * x.d };
}

template<typename T>
fn tangent<T> pown(auto x, tangent<T> n)
{
    using std::pow;

    return { pow(x, n.v), pow(x, n.v) * log(x) * n.d };
}

template<typename T>
fn tangent<T> pow(tangent<T> x, auto n)
{
    return pown(x, n);
}

template<typename T>
fn tangent<T> pow(auto x, tangent<T> n)
{
    return pown(x, n);
}

template<typename T>
fn tangent<T> pow(tangent<T> x, tangent<T> n)
{
    using std::pow;

    return { pow(x.v, n.v),
             n.v * pow(x.v, n.v - 1) * x.d + pow(x.v, n.v) * log(x.v) * n.d };
}

template<typename T>
fn bool isinf(tangent<T> a)
{
    using std::isinf;

    return isinf(a.v);
}

#undef fn

} // namespace cu

#endif // CUTANGENT_ARITHMETIC_BASIC_CUH
