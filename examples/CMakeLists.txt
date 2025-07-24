# Add an example with the given name and optional cuda flags.
# Usage:
#        add_example(name)
#        add_example(name cuda_flag1 cuda_flag2 ...)
function(add_example name)
    add_executable(${name} ${name}.cu)
    target_compile_features(${name} PRIVATE cxx_std_20 cuda_std_20)
    set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES native)
    target_link_libraries(${name} PRIVATE ${PROJECT_NAME} warnings)
    foreach(cuda_flag ${ARGN})
        target_compile_options(${name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${cuda_flag}>")
    endforeach()
endfunction()

add_subdirectory(basic)
add_subdirectory(interval)
add_subdirectory(mccormick)
add_subdirectory(finance)
add_subdirectory(formatting)
