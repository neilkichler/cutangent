#include "tests_against_finite_differences.h"
#include "tests_common.h"

int main()
{
    cu::test::device::init();
    int status = test_against_finite_differences();
    cu::test::device::reset();
    return status;
}
