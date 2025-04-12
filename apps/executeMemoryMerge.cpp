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


#include "../src/merge/merge.hpp"
#include "../src/utils/fileUtils.h"


#define DUPLICATION_FACTOR 2
#define PARTITION_NUM 8
#define MERGE_DEG 32


void mergeIndex(const std::string indexName, std::string baseFolder,
        const std::string& datasetPath,
        uint32_t folderNum,
        std::vector<std::vector<uint32_t>>& merged_index,
        bool isGPU = 0){

    uint32_t dataset_header[2];
    readMetadata(datasetPath, dataset_header);
    uint32_t dataset_size = dataset_header[0];
    printf("Dataset size: %d\n", dataset_size);

    omp_lock_t* locks = new omp_lock_t[dataset_size];
    for (int i = 0; i < dataset_size; i++) {
        omp_init_lock(&locks[i]);
    }
    merged_index.clear();
    merged_index.resize(dataset_size); // {global_index, global_neighbors}
    printf("generated locks and merged index\n");

    auto startTime = std::chrono::high_resolution_clock::now(); 

    uint32_t deg = MERGE_DEG;
    initGpuLocks();
    initEvents();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < folderNum; ++i) {
        std::filesystem::path subFolder = "partition" + std::to_string(i);
        std::filesystem::path IndexPath = std::filesystem::path(baseFolder) / subFolder / std::filesystem::path("index") / indexName;
        if (std::filesystem::exists(IndexPath) && std::filesystem::is_regular_file(IndexPath)) {
            std::cout << "Found matching file: " << IndexPath << std::endl;

            std::vector<std::vector<uint32_t>> index;
            readIndex(IndexPath.string(), index);


            std::string idx_file = baseFolder + "/partition" + std::to_string(i) + "/idmap.ibin";
            uint32_t header[1];
            readMetadataOneDimension(idx_file, header);
            std::vector<uint32_t> idx_vec(header[0]);
            readFileOneDimension<uint32_t>(idx_file, idx_vec); 

            printf("Read index and idx map\n");
            

            if (isGPU) {
                uint32_t gpu_id = 0;
                gpu_id = i % GPU_NUM;
                printf("Merge index shard %d using GPU %d\n", i, gpu_id);
                mergeShardAfterTranslationGPU(locks, merged_index, index, idx_vec, gpu_id);
            } 
            else {
                printf("Merge index shard %d using CPU\n", i);
                mergeShardAfterTranslation(locks, merged_index, index, idx_vec);
            }

            printf("merged index of one shard\n");

        } else {
            std::cout << "File not found: " << IndexPath << std::endl;
        }
    }
    destroyEvents();
    destroyGpuLocks();

    printf("Start Assigning!!\n");
    auto assignTime = std::chrono::high_resolution_clock::now();
    auto assignDuration = std::chrono::duration_cast<std::chrono::milliseconds>(assignTime - startTime);
 

    standardizeNeighborList(merged_index, deg);

    auto standardizeTime = std::chrono::high_resolution_clock::now();
    auto standardizeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(standardizeTime- assignTime);
 
    printf("assign time and standardize time are: %lld ms, %lld ms\n", assignDuration.count(), standardizeDuration.count());

    for (int i = 0; i < dataset_size; i++) {
    omp_destroy_lock(&locks[i]);
    }
    delete[] locks;

}



void mergeAllIndexInFolder(std::string baseFolder,
    std::string datasetPath){
    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<std::filesystem::path> subfolders;
    for (const auto& entry : std::filesystem::directory_iterator(baseFolder)) {
        if (entry.is_directory() && entry.path().filename().string().find("partition")==0) {
            subfolders.push_back(entry.path());
        }
    }

    std::filesystem::path firstSubfolder = subfolders[0];
    std::filesystem::path firstIndexFolder = firstSubfolder / std::filesystem::path("index");
    uint32_t folderNum = subfolders.size();

    auto prepareTime = std::chrono::high_resolution_clock::now();
    auto prepareDuration = std::chrono::duration_cast<std::chrono::milliseconds>(prepareTime - startTime);
    printf("prepare duration: %lld milliseconds\n", prepareDuration.count());

    auto lastIndexWrite = prepareTime;
    long long totalIndexMerge = 0;
    long long totalIndexWrite = 0;
    uint32_t iter_count = 0;
    for (const auto& file : std::filesystem::directory_iterator(firstIndexFolder)) {

        if (file.is_regular_file()) {

            std::string indexPath = file.path().string();
            std::string indexName = file.path().filename().string();
            
            std::vector<std::vector<uint32_t>> mergedIndex;
            // mergeIndex(indexName, idxMapFolder, datasetPath, subfolders, mergedIndex, 1);
            mergeIndex(indexName, baseFolder, datasetPath, subfolders.size(), mergedIndex, 0);

            printf("Index dimension: %d %d\n", mergedIndex.size(), mergedIndex[0].size());
            auto indexMergeTime = std::chrono::high_resolution_clock::now();
            auto indexMergeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(indexMergeTime - lastIndexWrite );
            printf("index %d merge duration: %lld milliseconds\n", iter_count, indexMergeDuration.count());


            const std::string index_file = baseFolder + "/mergedIndex/" + indexName;
            writeIndexMerged(index_file, mergedIndex);
            auto indexWriteTime = std::chrono::high_resolution_clock::now();
            auto indexWriteDuration = std::chrono::duration_cast<std::chrono::milliseconds>(indexWriteTime - indexMergeTime);
            printf("index %d write duration: %lld milliseconds\n", iter_count, indexWriteDuration.count());
            lastIndexWrite = indexWriteTime;

            long long totalTimeOfThisIter = indexMergeDuration.count() + indexWriteDuration.count();
            printf("index %d total duration: %lld milliseconds\n", iter_count, totalTimeOfThisIter);
            totalIndexMerge += indexMergeDuration.count();
            totalIndexWrite += indexWriteDuration.count(); 
            iter_count++;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    printf("total merge, write duration: %lld, %lld milliseconds \n", totalIndexMerge, totalIndexWrite);
    printf("overall duration: %lld milliseconds\n", overallDuration.count());
}

int main() {
    // nvcc ../partition/partition.cpp ../partition/disk_partition.cpp ../partition/kmeans.cpp ../partition/kmeans.cu ../merge/merge.cpp ../merge/merge.cu ../utils/indexIO.cpp ../utils/datasetIO.cpp ../utils/distance.cpp gpuManagement.cpp scheduler.cpp -I/home/lanlu/raft/cpp/include/ -I/home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include -I/home/lanlu/miniconda3/envs/rapids_raft/include -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx -I/home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include -I/home/lanlu/raft/cpp/build/_deps/benchmark-src/include -lcudart -ldl -lbenchmark -lpthread -lfmt -L/home/lanlu/raft/cpp/build/_deps/benchmark-build/src -Xcompiler -fopenmp -o testMerge
    std::string baseFolder = "/home/lanlu/scaleGANN/dataset/sift100M/D32_N8_epsilon1.2/";
    std::string datasetPath = "/home/lanlu/scaleGANN/dataset/sift100M/base.100M.u8bin";
    mergeAllIndexInFolder(baseFolder, datasetPath);
    return 0;
}

