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

#include "../../src/merge/disk_merge.h"

#define MERGE_DEG 32
#define BUILD_DEG 32

void mergeDisk(std::string base_folder, const std::string merge_index_path,
        const uint64_t nshards, uint32_t max_degree,
        const std::string index_name){
    std::string output_index_file = merge_index_path + "/" + index_name;
    std::string medoids_file = merge_index_path + "/" + index_name + "_medoids.bin";
    std::string labels_to_medoids_file = merge_index_path + "/" + index_name + "_lable_to_medoids.bin";
    DiskANN_merge(base_folder, index_name, base_folder,
        nshards, max_degree,
        output_index_file, medoids_file, false,
        labels_to_medoids_file);
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
        if (entry.is_directory() && entry.path().filename().string().find("partition")==0 && entry.path().filename().string().find(".data")==std::string::npos) {
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
    std::string baseFolder = "/home/lanlu/scaleGANN/dataset/sift100M/DiskANN/";
    std::string mergeFolder = "/home/lanlu/scaleGANN/dataset/sift100M/DiskANN/mergedIndex";
    mergeDisk_allIndexInFolder(baseFolder, mergeFolder);
    return 0;
}


