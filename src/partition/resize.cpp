#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include <omp.h>
#include <chrono>

#include "partition.h"
#include "../utils/fileUtils.h"


size_t getFileSize(std::string filename){
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t fileSize = file.tellg();
    
    return fileSize;
}

uint32_t getFileNpts(std::string filename, uint32_t ndim){
    uint32_t suffixType = suffixToType(filename);
    uint32_t typeSize = 0;
    if (suffixType == 0 || suffixType == 1) typeSize = 4;
    else if (suffixType == 2) typeSize = 1;
    else {
        throw std::invalid_argument("Unknown file suffix: " + suffixType);
    }

    size_t fileSize = getFileSize(filename);
    uint32_t fileNpts = (uint32_t) (fileSize / (typeSize * ndim));

    return fileNpts;
}

template <typename T> 
uint32_t getIncreasedPartitionNum(std::string filename,
        uint32_t memGPU, uint32_t npts, uint32_t ndim, uint32_t degree){
    
    uint32_t partitionNum = get_partition_num<T>(memGPU, npts, ndim, degree);
    
    return partitionNum;
}

// uint32_t getDecreasedPartitionNum(){
// }


void fetchIdxMapFromFile(uint32_t index, std::string idxMapFolder,
        std::vector<std::vector<std::vector<uint32_t>>>& idx_map){
    std::string IdxMapPath = idxMapFolder + "idmap" + std::to_string(index) + ".ibin";
    readIdxMaps(IdxMapPath, idx_map);
}


void refineIdxMapAfterSplit(uint32_t index, std::string idxMapFolder, 
        std::vector<std::vector<std::vector<uint32_t>>>& idx_map){

    std::vector<std::vector<std::vector<uint32_t>>> global_idx_map(0);
    fetchIdxMapFromFile(index, idxMapFolder, global_idx_map);
    uint32_t global_map_size = global_idx_map[0].size();
    printf("Global idx map size: %d\n", global_map_size);

    for (auto& map_entry : idx_map) {
        uint32_t entry_size = map_entry.size();
        printf("Current local idx map size: %d\n", entry_size);

        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < entry_size; i++){
        // for (auto& local_entry : map_entry) { 
            auto& local_entry = map_entry[i];
            // if (local_entry.size() != 2) {
            //     throw std::runtime_error("Invalid entry in idx_map: each entry must have two elements, representing local and global index respectively.");
            // }
            uint32_t local_index = local_entry[0];
            uint32_t global_index = local_entry[1];
            // if (i != local_index){
            //     throw std::runtime_error("Invalid idx_map order.");
            // }

            // if (global_index >= global_map_size){
            //     throw std::runtime_error("Global idx map size should be large enough.");
            // }
            local_entry[1] = global_idx_map[0][global_index][1]; 
        }
    }

}

template <typename T> 
void splitN(uint32_t n, uint32_t memGPU, uint32_t npts, uint32_t ndim, uint32_t max_iters,
        const std::vector<std::vector<T>>& data,
        std::vector<std::vector<std::vector<T>>>& partitions,
        std::vector<std::vector<std::vector<uint32_t>>>& idx_map){

    partitions.resize(n);
    get_partitions(memGPU, npts, ndim, max_iters, n, data, partitions, idx_map);
}


int extractNumber(const std::string& filePath) {
    uint32_t lastSlashPos = filePath.find_last_of('/');
    std::string fileName = (lastSlashPos == std::string::npos) ? filePath : filePath.substr(lastSlashPos + 1);

    size_t numStart = fileName.find_first_of("0123456789");
    if (numStart == std::string::npos) {
        throw std::runtime_error("No numbers found in the file name.");
    }

    size_t numEnd = fileName.find_first_not_of("0123456789", numStart);
    std::string numberStr = fileName.substr(numStart, numEnd - numStart);
    return std::stoi(numberStr);
}


template <typename T> 
void resize(std::string filename, uint32_t memGPU, uint32_t ndim, uint32_t degree, uint32_t max_iters,
        std::string idxMapFolder, std::string baseFolder){
    
    uint32_t npts = getFileNpts(filename, ndim);
    uint32_t split_num = getIncreasedPartitionNum<T>(filename, memGPU, npts, ndim, degree);
    
    if(split_num > 1){
        std::vector<std::vector<T>> data = std::vector<std::vector<T>>(npts, std::vector<T>(ndim));
        readFile<T>(filename, data);

        std::vector<std::vector<std::vector<T>>> partitions(split_num);
        std::vector<std::vector<std::vector<uint32_t>>> idx_map(split_num);
        splitN(split_num, memGPU, npts, ndim, max_iters, data, partitions, idx_map);

        uint32_t indexFile = extractNumber(filename);
        refineIdxMapAfterSplit(indexFile, idxMapFolder, idx_map);

        printf("Staring writing resized information\n");
        baseFolder = baseFolder + "/resize_partition_" + std::to_string(indexFile) + "/";
        writeDatasetPartitions<T>(baseFolder, partitions);
        writeIdxMaps(baseFolder, idx_map);
    }

}

// nvcc resize.cpp partition.cpp kmeans.cpp kmeans.cu ../merge/merge.cpp ../utils/indexIO.cpp ../utils/datasetIO.cpp ../utils/distance.cpp -I/home/lanlu/raft/cpp/include/ -I/home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include -I/home/lanlu/miniconda3/envs/rapids_raft/include -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx -I/home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include -I/home/lanlu/raft/cpp/build/_deps/benchmark-src/include -lcudart -ldl -lbenchmark -lpthread -lfmt -L/home/lanlu/raft/cpp/build/_deps/benchmark-build/src -Xcompiler -fopenmp -o testSplit
int main(){
    auto startTime = std::chrono::high_resolution_clock::now();
    std::string filename = "/home/lanlu/scaleGANN/dataset/sift100M/partitions32deg/partition8.u8bin";
    uint32_t memGPU = 16;
    uint32_t ndim = 128;
    uint32_t degree = 32;
    uint32_t max_iters = 15; // Vamana is 10
    std::string idxMapFolder = "/home/lanlu/scaleGANN/dataset/sift100M/partitions32deg/";
    std::string baseFolder = "/home/lanlu/scaleGANN/dataset/sift100M/partitions32deg/";

    resize<uint8_t>(filename, memGPU, ndim, degree, max_iters, idxMapFolder, baseFolder);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto resizeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    printf("resize duration: %lld milliseconds \n", resizeDuration.count());
}