cmake_minimum_required(VERSION 3.25.2)

project(cutangent-single-header-test LANGUAGES CUDA)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cutangent.cuh
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/single-header.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../../tools/single-header.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running Python script to generate single-header library"
)

add_custom_target(single-header DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cutangent.cuh)

add_executable(project-header project_header.cu)
add_dependencies(project-header single-header)
target_compile_features(project-header PRIVATE cuda_std_20)
target_compile_options(project-header PUBLIC "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>")
set_target_properties(project-header PROPERTIES CUDA_ARCHITECTURES native)

add_library(cutangent-single-header INTERFACE IMPORTED)
target_include_directories(cutangent-single-header INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(system-header system_header.cu)
add_dependencies(system-header single-header)
target_link_libraries(system-header PRIVATE cutangent-single-header)
target_compile_features(system-header PRIVATE cuda_std_20)
target_compile_options(system-header PUBLIC "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>")
set_target_properties(system-header PROPERTIES CUDA_ARCHITECTURES native)
