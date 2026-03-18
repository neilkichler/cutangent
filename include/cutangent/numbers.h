#ifndef CUTANGENT_NUMBERS_H
#define CUTANGENT_NUMBERS_H

#include <cutangent/tangent.h>

#include <numbers>

// Explicit specialization of math constants is allowed for custom types.
// See https://eel.is/c++draft/numbers#math.constants-2.
namespace std::numbers
{

template<typename T>
inline constexpr cu::tangent<T> e_v<cu::tangent<T>> = e_v<T>;

template<typename T>
inline constexpr cu::tangent<T> log2e_v<cu::tangent<T>> = log2e_v<T>;

template<typename T>
inline constexpr cu::tangent<T> log10e_v<cu::tangent<T>> = log10e_v<T>;

template<typename T>
inline constexpr cu::tangent<T> pi_v<cu::tangent<T>> = pi_v<T>;

template<typename T>
inline constexpr cu::tangent<T> inv_pi_v<cu::tangent<T>> = inv_pi_v<T>;

template<typename T>
inline constexpr cu::tangent<T> inv_sqrtpi_v<cu::tangent<T>> = inv_sqrtpi_v<T>;

template<typename T>
inline constexpr cu::tangent<T> ln2_v<cu::tangent<T>> = ln2_v<T>;

template<typename T>
inline constexpr cu::tangent<T> ln10_v<cu::tangent<T>> = ln10_v<T>;

template<typename T>
inline constexpr cu::tangent<T> sqrt2_v<cu::tangent<T>> = sqrt2_v<T>;

template<typename T>
inline constexpr cu::tangent<T> sqrt3_v<cu::tangent<T>> = sqrt3_v<T>;

template<typename T>
inline constexpr cu::tangent<T> inv_sqrt3_v<cu::tangent<T>> = inv_sqrt3_v<T>;

template<typename T>
inline constexpr cu::tangent<T> egamma_v<cu::tangent<T>> = egamma_v<T>;

template<typename T>
inline constexpr cu::tangent<T> phi_v<cu::tangent<T>> = phi_v<T>;

} // namespace std::numbers

// In cu:: we provide access to all the standard math constants and some additional helpful ones.
namespace cu
{

using std::numbers::e_v;
using std::numbers::egamma_v;
using std::numbers::inv_pi_v;
using std::numbers::inv_sqrt3_v;
using std::numbers::inv_sqrtpi_v;
using std::numbers::ln10_v;
using std::numbers::ln2_v;
using std::numbers::log10e_v;
using std::numbers::log2e_v;
using std::numbers::phi_v;
using std::numbers::pi_v;
using std::numbers::sqrt2_v;
using std::numbers::sqrt3_v;

#ifndef CU_NUMBERS_PRIMARY_DEFINED
#define CU_NUMBERS_PRIMARY_DEFINED

template<typename T>
inline constexpr T pi_2_v;

template<typename T>
inline constexpr T tau_v;

#endif // CU_NUMBERS_PRIMARY_DEFINED

template<typename T>
inline constexpr tangent<T> pi_2_v<tangent<T>> = pi_2_v<T>;

template<typename T>
inline constexpr tangent<T> tau_v<tangent<T>> = tau_v<T>;

} // namespace cu

#endif // CUTANGENT_NUMBERS_H
