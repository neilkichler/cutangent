#ifndef CUTANGENT_VTANGENT_H
#define CUTANGENT_VTANGENT_H

#include <compare>
#include <concepts>

namespace cu
{

template<typename T, int N>
struct vtangent
{
    T v;     // value
    T ds[N]; // derivatives

    constexpr vtangent() = default;

    constexpr vtangent(auto value)
        : v { static_cast<T>(value) }
        , ds { nullptr }
    { }

    constexpr vtangent(auto value, auto derivative)
        : v { value }
        , ds { nullptr }
    { }


    int size() const noexcept { return N; }

    constexpr auto operator<=>(const vtangent &other) const noexcept { return v <=> other.v; }
    constexpr bool operator==(const vtangent &other) const noexcept { return v == other.v; }
};

template<typename T, int N>
constexpr T &value(vtangent<T, N> &x)
{
    return x.v;
}

// template<typename T, int N>
// constexpr T &derivative(vtangent<T, N> &x)
// {
//     return x.ds;
// }

} // namespace cu

#endif // CUTANGENT_VTANGENT_H
