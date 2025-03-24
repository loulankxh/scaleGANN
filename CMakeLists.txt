cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
    ScaleGANN
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp -Xcompiler=-mavx2 -Xcompiler=-mfma -Xcompiler=-msse4.2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")

include_directories(
    /home/lanlu/raft/cpp/include/
    /home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include
    /home/lanlu/miniconda3/envs/rapids_raft/include
    /home/lanlu/miniconda3/envs/rapids_raft/include/rapids
    /home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx
    /home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include
    /home/lanlu/raft/cpp/build/_deps/benchmark-src/include
    /home/lanlu/scaleGANN/DiskANN/include
)

link_directories(
    /home/lanlu/raft/cpp/build/_deps/benchmark-build/src
)


set(DISKANN_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/DiskANN/src)
set(DISKANN_SOURCES
abstract_data_store.cpp ann_exception.cpp disk_utils.cpp 
distance.cpp index.cpp in_mem_graph_store.cpp in_mem_data_store.cpp
linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
in_mem_data_store.cpp in_mem_graph_store.cpp
natural_number_set.cpp memory_mapper.cpp partition.cpp pq.cpp
pq_flash_index.cpp scratch.cpp logger.cpp utils.cpp filter_utils.cpp index_factory.cpp abstract_index.cpp pq_l2_distance.cpp pq_data_store.cpp)
list(TRANSFORM DISKANN_SOURCES PREPEND ${DISKANN_SRC_PATH}/)

set(SCALEGANN_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(PARTITION_SOURCES 
    partition/partition.cpp partition/disk_partition.cpp partition/kmeans.cpp partition/kmeans.cu
    merge/merge.cpp merge/merge.cu
    utils/indexIO.cpp utils/datasetIO.cpp utils/distance.cpp
    scheduler/scheduler.cpp scheduler/gpuManagement.cpp)
list(TRANSFORM PARTITION_SOURCES  PREPEND ${SCALEGANN_SRC_PATH}/)


set(MERGE_SOURCES 
    partition/partition.cpp partition/disk_partition.cpp partition/kmeans.cpp partition/kmeans.cu
    merge/merge.cpp merge/merge.cu
    utils/indexIO.cpp utils/datasetIO.cpp utils/distance.cpp
    scheduler/scheduler.cpp scheduler/gpuManagement.cpp)
list(TRANSFORM MERGE_SOURCES  PREPEND ${SCALEGANN_SRC_PATH}/)


set(SEARCH_SCALEGANN_SOURCES 
    search/searchScaleGANN.cpp search/search.cpp 
    utils/indexIO.cpp utils/datasetIO.cpp utils/distance.cpp
)
list(TRANSFORM SEARCH_SCALEGANN_SOURCES  PREPEND ${SCALEGANN_SRC_PATH}/)

add_executable(executePartition ${PARTITION_SOURCES} ${DISKANN_SOURCES})
add_executable(executeMerge ${MERGE_SOURCES} ${DISKANN_SOURCES})
add_executable(searchScaleGANN ${SEARCH_SCALEGANN_SOURCES})

# set_target_properties(my_program PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES "60"  # 设置 GPU 架构，例如 60 表示 sm_60
# )


# target_link_libraries(executePartition
#     PRIVATE
#     cudart
#     dl
#     benchmark
#     pthread
#     fmt
#     mkl_rt
#     m
# )
set(COMMON_LIBS
    cudart
    dl
    benchmark
    pthread
    fmt
    mkl_rt
    m
)
foreach(target IN LISTS executePartition executeMerge searchScaleGANN)
    target_link_libraries(${target} PRIVATE ${COMMON_LIBS})
endforeach()