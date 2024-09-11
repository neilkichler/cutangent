set(INCLUDE_PATH "${cutangent_SOURCE_DIR}/include")
set(LIB_PATH "${INCLUDE_PATH}/cutangent")

set(headers 
    "${LIB_PATH}/arithmetic/basic.cuh"
    "${LIB_PATH}/arithmetic/intrinsic.cuh"
    "${LIB_PATH}/cutangent.cuh"
    "${LIB_PATH}/tangent.h"
    "${LIB_PATH}/vtangent.h"
)

add_library(cutangent "main.cu" ${headers})

target_include_directories(
    cutangent PUBLIC $<BUILD_INTERFACE:${INCLUDE_PATH}>
                      $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

# C++20 is required for all users of the library
target_compile_features(cutangent PUBLIC cxx_std_20 cuda_std_20)

source_group(
    TREE "${INCLUDE_PATH}"
    PREFIX "Header Files"
    FILES ${headers})

FetchContent_Declare(
  PackageProject
  GIT_REPOSITORY https://github.com/TheLartians/PackageProject.cmake.git
  GIT_TAG v1.11.2
)
FetchContent_MakeAvailable(PackageProject)

packageProject(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    NAMESPACE ${PROJECT_NAME}
    BINARY_DIR ${PROJECT_BINARY_DIR}
    INCLUDE_DIR ${INCLUDE_PATH}
    INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    DEPENDENCIES cuinterval
)
