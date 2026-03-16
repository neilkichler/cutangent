#include <cutangent/cutangent.cuh>

#include <iostream>

int main()
{
    cu::tangent<float> x = 1.0;
    cu::tangent<float> y = 2.0;

    derivative(x) = 1.0; // seed
    auto z        = exp(x * y);

    std::cout << derivative(z) << '\n';

    return 0;
};
