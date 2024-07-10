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

    constexpr auto operator<=>(const tangent &) const = default;
};

} // namespace cu

#endif // CUTANGENT_TANGENT_H
