#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <variant>
#include <cassert>
#include <omp.h>
#include <mkl.h>

#include "../partition/partition.h"
#include "../partition/disk_partition.h"
#include "../utils/fileUtils.h"
#include "gpuManagement.h"


#define GPU_MEMORY 16
#define BUILD_DEG 32
#define DUPLICATION_FACTOR 2
#define PARTITION_NUM 8
#define EPSILON 2
#define MAX_ITERATION 15


void partitionDisk_kmeans(const std::string file_path, std::string baseFolder){
    uint32_t num_threads = omp_get_num_procs();
    printf("Using %d threads\n", num_threads);
    omp_set_num_threads(num_threads);
    mkl_set_num_threads(num_threads);

    auto startTime = std::chrono::high_resolution_clock::now();

    double sampling_rate = 0.05;
    uint32_t memGPU = GPU_MEMORY;
    uint32_t degree = BUILD_DEG;
    size_t k_base = DUPLICATION_FACTOR;
    uint32_t partition_number = PARTITION_NUM;
    uint32_t epsilon = EPSILON;
    uint32_t max_iters = MAX_ITERATION; // Vamana is 1


    uint32_t suffixType = suffixToType(file_path);
    if(suffixType == 0){ // float
        scaleGANN_partitions_with_ram_budget<float>(file_path, sampling_rate, memGPU,
            degree, baseFolder, k_base, partition_number, epsilon, max_iters);
    } else if (suffixType == 2) { // uint8_t
        scaleGANN_partitions_with_ram_budget<uint8_t>(file_path, sampling_rate, memGPU,
            degree, baseFolder, k_base, partition_number, epsilon, max_iters);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    printf("Disk partition duration: %lld milliseconds\n", overallDuration.count());
}


int main() {
    // nvcc ../partition/partition.cpp ../partition/disk_partition.cpp ../partition/kmeans.cpp ../partition/kmeans.cu ../merge/merge.cpp ../merge/merge.cu ../utils/indexIO.cpp ../utils/datasetIO.cpp ../utils/distance.cpp scheduler.cpp gpuManagement.cpp -I/home/lanlu/raft/cpp/include/ -I/home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include -I/home/lanlu/miniconda3/envs/rapids_raft/include -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx -I/home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include -I/home/lanlu/raft/cpp/build/_deps/benchmark-src/include -lcudart -ldl -lbenchmark -lpthread -lfmt -L/home/lanlu/raft/cpp/build/_deps/benchmark-build/src -Xcompiler -fopenmp -o testPartition
    std::string file_path = "/home/lanlu/scaleGANN/dataset/sift100M/base.100M.u8bin";
    std::string baseFolder = "/home/lanlu/scaleGANN/dataset/sift100M/D64_N8";
    partitionDisk_kmeans(file_path, baseFolder);
    return 0;
}

