#ifndef CUTANGENT_FORMAT_H
#define CUTANGENT_FORMAT_H

#include <cutangent/tangent.h>

#include <format>
#include <ostream>

namespace cu
{

template<typename T>
std::ostream &operator<<(std::ostream &os, tangent<T> x)
{
    return os << "{v: " << x.v << ", d: " << x.d << "}";
}

} // namespace cu

template<typename T>
struct std::formatter<cu::tangent<T>> : std::formatter<T>
{
    auto format(const cu::tangent<T> &x, std::format_context &ctx) const
    {
        auto out = ctx.out();

        out = std::format_to(out, "{{v: ");
        out = std::formatter<T>::format(x.v, ctx);
        out = std::format_to(out, ", d: ");
        out = std::formatter<T>::format(x.d, ctx);
        return std::format_to(out, "}}");
    }
};

#endif // CUTANGENT_FORMAT_H
