#ifndef CUTANGENT_ARITHMETIC_BASIC_CUH
#define CUTANGENT_ARITHMETIC_BASIC_CUH

#include <cutangent/tangent.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <type_traits>

namespace cu
{

template<typename T>
concept arithmetic = std::is_arithmetic_v<T>;

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
fn tangent<T> operator+(tangent<T> a, arithmetic auto b)
{
    return { a.v + b, a.d };
}

template<typename T>
fn tangent<T> operator+(arithmetic auto a, tangent<T> b)
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
fn tangent<T> operator-(tangent<T> a, arithmetic auto b)
{
    return { a.v - b, a.d };
}

template<typename T>
fn tangent<T> operator-(T a, tangent<T> b)
{
    return { a - b.v, -b.d };
}

template<typename T>
fn tangent<T> operator-(arithmetic auto a, tangent<T> b)
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
fn tangent<T> operator*(tangent<T> a, arithmetic auto b)
{
    return { a.v * b, a.d * b };
}

template<typename T>
fn tangent<T> operator*(arithmetic auto a, tangent<T> b)
{
    return b * a;
}

template<typename T>
fn tangent<T> operator/(tangent<T> a, tangent<T> b)
{
    return { a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v) };
}

template<typename T>
fn tangent<T> operator/(T a, tangent<T> b)
{
    return { a / b.v, (-a * b.d) / (b.v * b.v) };
}

template<typename T>
fn tangent<T> operator/(tangent<T> a, T b)
{
    return { a.v / b, a.d / b };
}

template<typename T>
fn tangent<T> operator/(arithmetic auto a, tangent<T> b)
{
    return { a / b.v, (-a * b.d) / (b.v * b.v) };
}

template<typename T>
fn tangent<T> operator/(tangent<T> a, arithmetic auto b)
{
    return { a.v / b, a.d / b };
}

template<typename T>
fn tangent<T> &operator+=(tangent<T> &a, auto b)
{
    a = a + b;
    return a;
}

template<typename T>
fn tangent<T> &operator-=(tangent<T> &a, auto b)
{
    a = a - b;
    return a;
}

template<typename T>
fn tangent<T> &operator*=(tangent<T> &a, auto b)
{
    a = a * b;
    return a;
}

template<typename T>
fn tangent<T> &operator/=(tangent<T> &a, auto b)
{
    a = a / b;
    return a;
}

template<typename T>
fn tangent<T> max(tangent<T> a, tangent<T> b)
{
    using std::max;

    T delta = a.v - b.v;
    if (delta < 0.0) {
        return { b.v, b.d };
    } else if (delta > 0.0) {
        return { a.v, a.d };
    } else {
        // many elements of the subdifferential could be chosen
        return { a.v, max(a.d, b.d) };
    }
}

template<typename T>
fn tangent<T> min(tangent<T> a, tangent<T> b)
{
    using std::min;

    T delta = a.v - b.v;
    if (delta < 0.0) {
        return { a.v, a.d };
    } else if (delta > 0.0) {
        return { b.v, b.d };
    } else {
        // many elements of the subdifferential could be chosen
        return { b.v, min(a.d, b.d) };
    }
}

template<typename T>
fn tangent<T> abs(tangent<T> x)
{
    using std::abs, std::copysign;

    // If the value of x is zero we take the sign of the directional derivative part
    // to match the more general notion of lexicographic differentiation as in
    // Example 4.2 of https://doi.org/10.1080/10556788.2015.1025400
    // and Section 6.2 of Griewanks stable piecewise linearizations:
    // https://doi.org/10.1080/10556788.2013.796683

    constexpr T zero {};
    T v = x.v == zero ? x.d : x.v;
    return { abs(x.v), copysign(1.0, v) * x.d };
}

template<typename T>
fn tangent<T> clamp(tangent<T> v, tangent<T> lb, tangent<T> ub)
{
    return max(lb, min(v, ub));
}

template<typename T>
fn tangent<T> mid(tangent<T> v, tangent<T> lb, tangent<T> ub)
{
    return clamp(v, lb, ub);
}

template<typename T>
fn tangent<T> recip(tangent<T> x)
{
    using std::pow;

    return { 1. / x.v, -x.d / pow(x.v, 2) };
}

template<typename T>
fn tangent<T> sin(tangent<T> x)
{
    using std::sin, std::cos;

    return { sin(x.v), cos(x.v) * x.d };
}

template<typename T>
fn tangent<T> cos(tangent<T> x)
{
    using std::cos, std::sin;

    return { cos(x.v), -sin(x.v) * x.d };
}

template<typename T>
fn tangent<T> tan(tangent<T> x)
{
    using std::pow, std::tan;

    return { tan(x.v), static_cast<T>(1.0) + pow(tan(x.v), 2) };
}

template<typename T>
fn tangent<T> asin(tangent<T> x)
{
    using std::asin, std::sqrt, std::pow;

    return { asin(x.v), x.d / sqrt(1.0 - pow(x.v, 2)) };
}

template<typename T>
fn tangent<T> acos(tangent<T> x)
{
    using std::acos, std::sqrt, std::pow;

    return { acos(x.v), -x.d / sqrt(1.0 - pow(x.v, 2)) };
}

template<typename T>
fn tangent<T> atan(tangent<T> x)
{
    using std::atan, std::pow;

    return { atan(x.v), x.d / (1.0 + pow(x.v, 2)) };
}

template<typename T>
fn tangent<T> atan2(tangent<T> a, tangent<T> b)
{
    using std::atan2, std::pow;

    return { atan2(a.v, b.v), (a.d * b.v - b.d * a.v) / (pow(a.v, 2) + pow(b.v, 2)) };
}

template<typename T>
fn tangent<T> atan2(tangent<T> a, T b)
{
    using std::atan2, std::pow;

    return { atan2(a.v, b), b * a.d / (pow(a.v, 2) + pow(b, 2)) };
}

template<typename T>
fn tangent<T> atan2(T a, tangent<T> b)
{
    using std::atan2, std::pow;

    return { atan2(a, b.v), -a * b.d / (pow(a, 2) + pow(b.v, 2)) };
}

template<typename T>
fn tangent<T> sinh(tangent<T> x)
{
    using std::sinh, std::cosh;

    return { sinh(x.v), cosh(x.v) * x.d };
}

template<typename T>
fn tangent<T> cosh(tangent<T> x)
{
    using std::cosh, std::sinh;

    return { cosh(x.v), sinh(x.v) * x.d };
}

template<typename T>
fn tangent<T> tanh(tangent<T> x)
{
    using std::tanh, std::pow, std::cosh;

    return { tanh(x.v), x.d / (pow(cosh(x.v), 2)) };
}

template<typename T>
fn tangent<T> asinh(tangent<T> x)
{
    using std::asinh, std::sqrt, std::pow;

    return { asinh(x.v), x.d / sqrt(pow(x.v, 2) + 1.0) };
}

template<typename T>
fn tangent<T> acosh(tangent<T> x)
{
    using std::acosh, std::sqrt, std::pow;

    return { acosh(x.v), x.d / sqrt(pow(x.v, 2) - 1.0) };
}

template<typename T>
fn tangent<T> atanh(tangent<T> x)
{
    using std::atanh, std::pow;

    return { atanh(x.v), x.d / (1.0 - pow(x.v, 2)) };
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

    return { log2(x.v), x.d / (x.v * std::numbers::ln2_v<T>)};
}

template<typename T>
fn tangent<T> log10(tangent<T> x)
{
    using std::log10;

    return { log10(x.v), x.d / (x.v * std::numbers::ln10_v<T>)};
}

template<typename T>
fn tangent<T> sqr(tangent<T> x)
{
    return { sqr(x.v), 2.0 * x.v * x.d };
}

template<typename T>
fn tangent<T> sqrt(tangent<T> x)
{
    using std::sqrt, std::numeric_limits;

    constexpr T zero {};
    // NOTE: We currently do not treat the case where x.v == 0, x.d > 0 to map to +inf.
    return { sqrt(x.v), x.d / (2.0 * sqrt(x.v) + (x.v == zero ? numeric_limits<T>::min() : zero)) };
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
    using std::pow, std::log;

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
    using ::isinf, std::isinf;

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

template<typename T>
fn tangent<T> erf(tangent<T> x)
{
    using std::erf, std::exp, std::pow, std::sqrt;

    return { erf(x.v), 2.0 * x.d * exp(-pow(x.v, 2)) / sqrt(std::numbers::pi) };
}

template<typename T>
fn tangent<T> erfc(tangent<T> x)
{
    using std::erfc, std::exp, std::pow, std::sqrt;

    return { erfc(x.v), -2.0 * x.d * exp(-pow(x.v, 2)) / sqrt(std::numbers::pi) };
}

template<typename T>
fn bool operator>(tangent<T> x, auto y)
{
    return x.v > y;
}

template<typename T>
fn bool operator<(tangent<T> x, auto y)
{
    return x.v < y;
}

#undef fn

} // namespace cu

#endif // CUTANGENT_ARITHMETIC_BASIC_CUH
