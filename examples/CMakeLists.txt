set (sample_naive_softmax_files
    navie_softmax.cu
)

set (sample_online_softmax_files
    sample_online_softmax.cu
)

set (SAMPLES_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/online_softmax

)

include_directories(
    ${SAMPLES_HEADER_DIR}
)

link_directories(
    ${PROJECT_SOURCE_DIR}/online_softmax/lib
)

message("-- Assign include directories (include_directories=${SAMPLES_HEADER_DIR})")

add_executable(navie_softmax ${sample_naive_softmax_files})
target_compile_features(navie_softmax PUBLIC cxx_std_14)
target_link_libraries(navie_softmax PUBLIC softmax)

add_executable(online_softmax ${sample_online_softmax_files})
target_compile_features(online_softmax PUBLIC cxx_std_14)
target_link_libraries(online_softmax PUBLIC softmax)