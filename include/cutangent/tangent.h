#ifndef CUTANGENT_TANGENT_H
#define CUTANGENT_TANGENT_H

#include <compare>

namespace cu
{

template<typename T>
struct tangent
{
    T v; // value
    T d; // derivative

    constexpr tangent() = default;

    constexpr tangent(auto value)
        : v { static_cast<T>(value) }
        , d { static_cast<T>(0) }
    { }

    constexpr tangent(auto value, auto derivative)
        : v { value }
        , d { derivative }
    { }

    constexpr auto operator<=>(const tangent &other) const noexcept { return v <=> other.v; }
    constexpr bool operator==(const tangent &other) const noexcept { return v == other.v; }
};

template<typename T>
constexpr T &value(tangent<T> &x)
{
    return x.v;
}

template<typename T>
constexpr T &derivative(tangent<T> &x)
{
    return x.d;
}

} // namespace cu

#endif // CUTANGENT_TANGENT_H
