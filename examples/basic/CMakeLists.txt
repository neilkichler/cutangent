add_executable(basic basic.cu)
target_compile_features(basic PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(basic PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(basic PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
target_link_libraries(basic PRIVATE ${PROJECT_NAME})

