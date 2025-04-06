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

#include "../src/merge/disk_merge.h"

#define MERGE_DEG 32
#define BUILD_DEG 32

void mergeDisk(std::string base_folder, const std::string merge_index_path,
        const uint64_t nshards, uint32_t merge_degree,
        const std::string index_name){
    std::string output_index_file = merge_index_path + "/" + index_name;
    scaleGANN_merge(base_folder,
                nshards, merge_degree, BUILD_DEG,
                output_index_file,
                index_name);
}

// There can be multiple index of different construction parameters in a folder
void mergeDisk_allIndexInFolder(const std::string baseFolder, const std::string mergeIndexPath){
    
    auto startTime = std::chrono::high_resolution_clock::now();

    if (!std::filesystem::exists(mergeIndexPath)) {
        if (std::filesystem::create_directories(mergeIndexPath)) {
            std::cout << "Dir doesn't exists but success to create such dir: " << mergeIndexPath << std::endl;
        } else {
            std::cerr << "Dir doesn't exists and fail to create such dir: " << mergeIndexPath << std::endl;
        }
    }

    std::vector<std::filesystem::path> subfolders;
    for (const auto& entry : std::filesystem::directory_iterator(baseFolder)) {
        if (entry.is_directory() && entry.path().filename().string().find("partition")==0) {
            subfolders.push_back(entry.path());
        }
    }
    std::filesystem::path firstSubfolder = subfolders[0];
    std::filesystem::path firstIndexFolder = firstSubfolder / std::filesystem::path("index");
    uint32_t shardNum = subfolders.size();

    auto prepareTime = std::chrono::high_resolution_clock::now();
    auto prepareDuration = std::chrono::duration_cast<std::chrono::milliseconds>(prepareTime - startTime);
    printf("prepare duration: %lld milliseconds\n", prepareDuration.count());

    auto lastIndexMerge = prepareTime;
    long long totalIndexMerge = 0;
    uint32_t iter_count = 0;
    for (const auto& file : std::filesystem::directory_iterator(firstIndexFolder)) {

        if (file.is_regular_file()) {

            std::string indexPath = file.path().string();
            std::string indexName = file.path().filename().string();
            
            std::vector<std::vector<uint32_t>> mergedIndex;
            mergeDisk(baseFolder, mergeIndexPath, shardNum, MERGE_DEG, indexName);

            auto indexMergeTime = std::chrono::high_resolution_clock::now();
            auto indexMergeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(indexMergeTime - lastIndexMerge);
            printf("index %d merge & write duration: %lld milliseconds\n", iter_count, indexMergeDuration.count());

            
            totalIndexMerge += indexMergeDuration.count();
            lastIndexMerge = indexMergeTime;
            iter_count++;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    printf("total merge duration: %lld milliseconds \n", totalIndexMerge);
    printf("overall duration: %lld milliseconds\n", overallDuration.count());
}

int main() {
    // nvcc ../partition/partition.cpp ../partition/disk_partition.cpp ../partition/kmeans.cpp ../partition/kmeans.cu ../merge/merge.cpp ../merge/merge.cu ../utils/indexIO.cpp ../utils/datasetIO.cpp ../utils/distance.cpp gpuManagement.cpp scheduler.cpp -I/home/lanlu/raft/cpp/include/ -I/home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include -I/home/lanlu/miniconda3/envs/rapids_raft/include -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx -I/home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include -I/home/lanlu/raft/cpp/build/_deps/benchmark-src/include -lcudart -ldl -lbenchmark -lpthread -lfmt -L/home/lanlu/raft/cpp/build/_deps/benchmark-build/src -Xcompiler -fopenmp -o testMerge
    std::string baseFolder = "/home/lanlu/scaleGANN/dataset/sift100M/D32_N8_epsilon2/";
    std::string mergeFolder = "/home/lanlu/scaleGANN/dataset/sift100M/D32_N8_epsilon2/mergedIndex";
    mergeDisk_allIndexInFolder(baseFolder, mergeFolder);
    return 0;
}


