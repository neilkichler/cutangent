#include <cutangent/cutangent.cuh>

// We do not put the formatting functionality in the default header because
// some might choose to use a different formatting. It does support ostream
// and std::format based output.
//
// The std::format accepts a format specifier that supports all the
// specifiers that exist for the underlying type T of cu::tangent<T>.
#include <cutangent/format.h>

#include <iostream>

int main()
{
    cu::tangent<double> x;
    value(x)      = 2.0;
    derivative(x) = 1.0;

    std::cout << x << '\n';
    std::cout << std::format("{}\n", x);
    // in c++23 you may use std::print and std::println
    // std::println("{}", x);

    // type specifiers like for double (applies to both value and derivative)
    std::cout << std::format("{:8.4f}\n", x);
    return 0;
}
