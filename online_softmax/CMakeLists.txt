set(softmax_files
    online_softmax.cu
)

set (SOFTMAX_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/online_softmax
    
)

include_directories(
    ${SOFTMAX_HEADER_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/online_softmax/lib)

message("-- Assign sgemv include directories (include_directories=${SOFTMAX_HEADER_DIR})")
message("-- in sgemv cuda Assign arch (arch=${CMAKE_CUDA_ARCHITECTURES})")

add_library(softmax SHARED ${softmax_files})
# Request that particles be built with --std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(softmax PUBLIC cxx_std_14)
target_compile_options(softmax PUBLIC "-gencode=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(softmax PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(softmax PUBLIC -lcudart -lcublas)

