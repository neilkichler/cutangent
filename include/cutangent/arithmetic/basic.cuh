#ifndef CUTANGENT_ARITHMETIC_BASIC_CUH
#define CUTANGENT_ARITHMETIC_BASIC_CUH

#include <cutangent/tangent.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <numbers>

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
fn tangent<T> operator+(tangent<T> a, T b)
{
    return { a.v + b, a.d };
}

template<typename T>
fn tangent<T> operator+(T a, tangent<T> b)
{
    return { a + b.v, b.d };
}

template<typename T>
fn tangent<T> operator-(tangent<T> a, tangent<T> b)
{
    return { a.v - b.v, a.d - b.d };
}

template<typename T>
fn tangent<T> operator-(tangent<T> a, T b)
{
    return { a.v - b, a.d };
}

template<typename T>
fn tangent<T> operator-(T a, tangent<T> b)
{
    return { a - b.v, -b.d };
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
fn tangent<T> operator*(tangent<T> a, std::integral auto b)
{
    return { a.v * static_cast<T>(b), a.d * static_cast<T>(b) };
}

template<typename T>
fn tangent<T> operator*(std::integral auto a, tangent<T> b)
{
    return b * a;
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
fn tangent<T> tan(tangent<T> x)
{
    using std::tan;

    return { tan(x.v), static_cast<T>(1.0) + sqr(tan(x.v)) };
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
fn tangent<T> log2(tangent<T> x)
{
    using std::log2;

    return { log2(x.v), x.d / (x.v * std::numbers::ln2_v<T>) };
}

template<typename T>
fn tangent<T> log10(tangent<T> x)
{
    using std::log10;

    return { log10(x.v), x.d / (x.v * std::numbers::ln10_v<T>) };
}

template<typename T>
fn tangent<T> sqr(tangent<T> x)
{
    return { sqr(x.v), 2.0 * x.v * x.d };
}

template<typename T>
fn tangent<T> sqrt(tangent<T> x)
{
    using std::numeric_limits;
    using std::sqrt;
    // NOTE: We currently do not treat the case where x.v == 0, x.d > 0 to map to +inf.
    return { sqrt(x.v), x.d / (2.0 * sqrt(x.v) + (x.v == static_cast<T>(0.0) ? numeric_limits<T>::min() : 0.0)) };
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
    using std::log;
    using std::pow;

    return { pow(x.v, n.v),
             n.v * pow(x.v, n.v - 1) * x.d + pow(x.v, n.v) * log(x.v) * n.d };
}

template<typename T>
fn tangent<T> cbrt(tangent<T> x)
{
    using std::cbrt;

    return { cbrt(x.v), x.d / (static_cast<T>(3.0) * sqr(cbrt(x.v))) };
}

template<typename T>
fn tangent<T> ceil(tangent<T> x)
{
    using std::ceil;

    return { ceil(x.v), 0.0 };
}

template<typename T>
fn T rint(tangent<T> x)
{
    using std::rint;

    return rint(x.v);
}

template<typename T>
fn long lrint(tangent<T> x)
{
    using std::lrint;

    return lrint(x.v);
}

template<typename T>
fn long long llrint(tangent<T> x)
{
    using std::llrint;

    return llrint(x.v);
}

template<typename T>
fn tangent<T> floor(tangent<T> x)
{
    using std::floor;

    return { floor(x.v), 0.0 };
}

template<typename T>
fn bool isinf(tangent<T> x)
{
    using std::isinf;

    return isinf(x.v);
}

template<typename T>
fn bool isfinite(tangent<T> x)
{
    using std::isfinite;

    return isfinite(x.v);
}

template<typename T>
fn bool isnan(tangent<T> a)
{
    using std::isnan;

    return isnan(a.v);
}

template<typename T>
fn tangent<T> remquo(tangent<T> x, tangent<T> y, int *quo)
{
    using std::remquo;

    return { remquo(x.v, y.v, quo), 0.0 };
}

template<typename T>
fn bool signbit(tangent<T> x)
{
    using std::signbit;

    return signbit(x.v);
}

template<typename T>
fn tangent<T> copysign(tangent<T> mag, T sgn)
{
    using std::copysign;

    return { copysign(mag.v, sgn), copysign(mag.d, sgn) };
}

#undef fn

} // namespace cu

#endif // CUTANGENT_ARITHMETIC_BASIC_CUH
