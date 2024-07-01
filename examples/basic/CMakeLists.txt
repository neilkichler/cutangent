add_executable(basic basic.cu)

include(FetchContent)
FetchContent_Declare(
  cumccormick
  GIT_REPOSITORY https://github.com/neilkichler/cumccormick.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cumccormick)

target_link_libraries(basic PUBLIC cumccormick)

target_compile_features(basic PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(basic PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(basic PRIVATE ${PROJECT_NAME})
