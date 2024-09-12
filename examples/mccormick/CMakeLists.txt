include(FetchContent)
FetchContent_Declare(
  cumccormick
  GIT_REPOSITORY https://github.com/neilkichler/cumccormick.git
  GIT_TAG main
)
FetchContent_MakeAvailable(cumccormick)

add_executable(mccormick mccormick.cu)
target_link_libraries(mccormick PUBLIC cumccormick)
target_compile_features(mccormick PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(mccormick PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(mccormick PRIVATE ${PROJECT_NAME})

add_executable(mccormick_v mccormick_v.cu)
target_link_libraries(mccormick_v PUBLIC cumccormick)
target_compile_features(mccormick_v PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(mccormick_v PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(mccormick_v PRIVATE ${PROJECT_NAME})
