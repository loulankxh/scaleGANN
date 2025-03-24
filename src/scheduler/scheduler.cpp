#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <variant>
#include <cassert>
#include <omp.h>
// #include <boost/program_options.hpp>
#include <mkl.h>

#include "../partition/partition.h"
#include "../partition/disk_partition.h"
#include "../merge/merge.hpp"
#include "../utils/fileUtils.h"
#include "gpuManagement.h"


#define DUPLICATION_FACTOR 2
#define PARTITION_NUM 8
#define BUILD_DEG 32
#define MERGE_DEG 32


// namespace po = boost::program_options;

void partitionDisk_kmeans(const std::string file_path, std::string baseFolder){
    uint32_t num_threads = omp_get_num_procs();
    printf("Using %d threads\n", num_threads);
    omp_set_num_threads(num_threads);
    mkl_set_num_threads(num_threads);
    auto startTime = std::chrono::high_resolution_clock::now();
    uint32_t memGPU = 16;
    uint32_t degree = BUILD_DEG;
    uint32_t max_iters = 15; // Vamana is 10
    double sampling_rate = 0.05;
    size_t k_base = 2;
    uint32_t epsilon = 2;
    scaleGANN_partitions_with_ram_budget<uint8_t>(file_path, sampling_rate, memGPU,
    degree, baseFolder, k_base, PARTITION_NUM, epsilon);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    printf("Disk partition duration: %lld milliseconds\n", overallDuration.count());
}



void partitionDataset_kmeans(const std::string file_path, std::string baseFolder){
    auto startTime = std::chrono::high_resolution_clock::now();

    // std::string file_path = "../../../python/raft-ann-bench/src/datasets/deep-image-96-inner/base_base.fbin";
    uint32_t suffixType = suffixToType(file_path);
    
    uint32_t header[2];
    readMetadata(file_path, header);
    uint32_t npts = header[0];
    uint32_t ndim = header[1];
    printf("Testing.... Rows & Dims: %d, %d\n", npts, ndim);

    std::variant<std::vector<std::vector<float>>, 
            std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint8_t>>> vec;

    if(suffixType == 0){ // float
        vec = std::vector<std::vector<float>>(npts, std::vector<float>(ndim));
        try {
            auto& floatVec = std::get<std::vector<std::vector<float>>>(vec);
            readFile<float>(file_path, floatVec);
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    } else if (suffixType == 1) { // uint32_t
        vec = std::vector<std::vector<uint32_t>>(npts, std::vector<uint32_t>(ndim));
        try {
            auto& uint32Vec = std::get<std::vector<std::vector<uint32_t>>>(vec);
            readFile<uint32_t>(file_path, uint32Vec);
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
   } else if (suffixType == 2) { // uint8_t
        vec = std::vector<std::vector<uint8_t>>(npts, std::vector<uint8_t>(ndim));
        try {
            auto& uint8Vec = std::get<std::vector<std::vector<uint8_t>>>(vec);
            readFile<uint8_t>(file_path, uint8Vec);
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
   }
    auto readTime = std::chrono::high_resolution_clock::now();

    uint32_t memGPU = 16;
    uint32_t degree = 32;
    uint32_t max_iters = 15; // Vamana is 10
    // std::vector<std::vector<std::vector<uint32_t>>> idx_map(0);
    std::vector<std::vector<uint32_t>> idx_map(0);
    std::variant<std::vector<std::vector<std::vector<float>>>,
                std::vector<std::vector<std::vector<uint32_t>>>, std::vector<std::vector<std::vector<uint8_t>>>> partitions;
    if(suffixType == 0){ // float
        try {
            auto& floatVec = std::get<std::vector<std::vector<float>>>(vec);
            partitions = main_partitions<float>(memGPU, npts, ndim, degree * 2 / 3, max_iters, DUPLICATION_FACTOR, PARTITION_NUM, floatVec, idx_map);
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    } else if (suffixType == 1) { // uint32_t
        try {
            auto& uint32Vec = std::get<std::vector<std::vector<uint32_t>>>(vec);
            partitions = main_partitions<uint32_t>(memGPU, npts, ndim, degree * 2 / 3, max_iters, DUPLICATION_FACTOR, PARTITION_NUM, uint32Vec, idx_map);
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    } else if (suffixType == 2) { // uint8_t
        try {
            auto& uint8Vec = std::get<std::vector<std::vector<uint8_t>>>(vec);
            partitions = main_partitions<uint8_t>(memGPU, npts, ndim, degree * 2 / 3, max_iters, DUPLICATION_FACTOR, PARTITION_NUM, uint8Vec, idx_map);
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

    }
    auto partitionTime = std::chrono::high_resolution_clock::now();

    // std::string baseFolder = "../../dataset/";
    if(suffixType == 0){ // float
        try {
            auto& floatPartitions = std::get<std::vector<std::vector<std::vector<float>>>>(partitions);
            writeDatasetPartitions<float>(baseFolder, floatPartitions);
            if(floatPartitions.size()>1){
                writeIdxMaps(baseFolder, idx_map);
            }
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    } else if (suffixType == 1) { // uint32_t
        try {
            auto& uint32Partitions = std::get<std::vector<std::vector<std::vector<uint32_t>>>>(partitions);
            writeDatasetPartitions<uint32_t>(baseFolder, uint32Partitions);
            if(uint32Partitions.size()>1){
                writeIdxMaps(baseFolder, idx_map);
            }
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    } else if (suffixType == 2) { // uint8_t
        try {
            auto& uint8Partitions = std::get<std::vector<std::vector<std::vector<uint8_t>>>>(partitions);
            writeDatasetPartitions<uint8_t>(baseFolder, uint8Partitions);
            if(uint8Partitions.size()>1){
                writeIdxMaps(baseFolder, idx_map);
            }
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    auto writeTime = std::chrono::high_resolution_clock::now();
    auto readDuration = std::chrono::duration_cast<std::chrono::milliseconds>(readTime - startTime);
    auto partitionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(partitionTime - readTime);
    auto writeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(writeTime - partitionTime);
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(writeTime - startTime);
    printf("read, partition, write duration: %lld, %lld, %lld milliseconds \n", readDuration.count(), partitionDuration.count(), writeDuration.count());
    printf("Kmeans Partition takes overall duration: %lld milliseconds\n", overallDuration.count());
}



void partitionDataset_direct(std::string file_path, std::string baseFolder){
    auto startTime = std::chrono::high_resolution_clock::now();

    uint32_t suffixType = suffixToType(file_path);
    
    uint32_t header[2];
    readMetadata(file_path, header);
    uint32_t npts = header[0];
    uint32_t ndim = header[1];
    printf("Testing.... Rows & Dims: %d, %d\n", npts, ndim);

    uint32_t memGPU = 16;
    uint32_t degree = 32;
    uint32_t partition_num = PARTITION_NUM;

    if(suffixType == 0){ // float
        std::vector<std::vector<float>> dataVec(npts, std::vector<float>(ndim));
        readFile<float>(file_path, dataVec);

        uint32_t partition_lower_bound = get_partition_num<float>(memGPU, npts, ndim, degree, 1);
        if (partition_lower_bound > partition_num) {
            printf("Needing %d partitions at least, change given partition number %d to %d\n", partition_lower_bound, partition_num, partition_lower_bound);
            partition_num = partition_lower_bound;
        }
        uint32_t each_partition_size = npts / partition_num;
        uint32_t last_partition_size = each_partition_size + npts % partition_num;

        std::vector<std::vector<std::vector<float>>> partitions;
        for(uint32_t iter = 0; iter < partition_num; iter ++){
            std::vector<std::vector<float>> segmented;
            size_t offset = iter * each_partition_size;
            if (iter < partition_num - 1){
                segmented.insert(segmented.end(), dataVec.begin() + offset, dataVec.begin() + offset + each_partition_size);
            } else{
                segmented.insert(segmented.end(), dataVec.begin() + offset, dataVec.begin() + offset + last_partition_size);

            }
            partitions.push_back(segmented);
        }

        writeDatasetPartitions<float>(baseFolder, partitions);
        
    } else if (suffixType == 1) { // uint32_t
        std::vector<std::vector<uint32_t>> dataVec(npts, std::vector<uint32_t>(ndim));
        readFile<uint32_t>(file_path, dataVec);

        uint32_t partition_lower_bound = get_partition_num<uint32_t>(memGPU, npts, ndim, degree, 1);
        if (partition_lower_bound > partition_num) {
            printf("Needing %d partitions at least, change given partition number %d to %d\n", partition_lower_bound, partition_num, partition_lower_bound);
            partition_num = partition_lower_bound;
        }
        uint32_t each_partition_size = npts / partition_num;
        uint32_t last_partition_size = each_partition_size + npts % partition_num;

        std::vector<std::vector<std::vector<uint32_t>>> partitions;
        for(uint32_t iter = 0; iter < partition_num; iter ++){
            std::vector<std::vector<uint32_t>> segmented;
            size_t offset = iter * each_partition_size;
            if (iter < partition_num - 1){
                segmented.insert(segmented.end(), dataVec.begin() + offset, dataVec.begin() + offset + each_partition_size);
            } else{
                segmented.insert(segmented.end(), dataVec.begin() + offset, dataVec.begin() + offset + last_partition_size);

            }
            partitions.push_back(segmented);
        }

        writeDatasetPartitions<uint32_t>(baseFolder, partitions);
   } else if (suffixType == 2) { // uint8_t
        std::vector<std::vector<uint8_t>> dataVec(npts, std::vector<uint8_t>(ndim));
        readFile<uint8_t>(file_path, dataVec);

        uint32_t partition_lower_bound = get_partition_num<uint8_t>(memGPU, npts, ndim, degree, 1);
        if (partition_lower_bound > partition_num) {
            printf("Needing %d partitions at least, change given partition number %d to %d\n", partition_lower_bound, partition_num, partition_lower_bound);
            partition_num = partition_lower_bound;
        }
        uint32_t each_partition_size = npts / partition_num;
        uint32_t last_partition_size = each_partition_size + npts % partition_num;

        std::vector<std::vector<std::vector<uint8_t>>> partitions;
        for(uint32_t iter = 0; iter < partition_num; iter ++){
            std::vector<std::vector<uint8_t>> segmented;
            size_t offset = iter * each_partition_size;
            if (iter < partition_num - 1){
                segmented.insert(segmented.end(), dataVec.begin() + offset, dataVec.begin() + offset + each_partition_size);
            } else{
                segmented.insert(segmented.end(), dataVec.begin() + offset, dataVec.begin() + offset + last_partition_size);

            }
            partitions.push_back(segmented);
        }

        writeDatasetPartitions<uint8_t>(baseFolder, partitions);
   }

   auto endTime = std::chrono::high_resolution_clock::now();
   auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
   printf("Direct CAGRA partition takes overall duration: %lld milliseconds\n", overallDuration.count());
}



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
    // nvcc ../partition/partition.cpp ../partition/disk_partition.cpp ../partition/kmeans.cpp ../partition/kmeans.cu ../merge/merge.cpp ../merge/merge.cu ../utils/indexIO.cpp ../utils/datasetIO.cpp ../utils/distance.cpp scheduler.cpp gpuManagement.cpp -I/home/lanlu/raft/cpp/include/ -I/home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include -I/home/lanlu/miniconda3/envs/rapids_raft/include -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx -I/home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include -I/home/lanlu/raft/cpp/build/_deps/benchmark-src/include -lcudart -ldl -lbenchmark -lpthread -lfmt -L/home/lanlu/raft/cpp/build/_deps/benchmark-build/src -Xcompiler -fopenmp -o testPartition
    // std::string file_path = "/home/lanlu/scaleGANN/dataset/sift100M/base.100M.u8bin";
    // std::string baseFolder = "/home/lanlu/scaleGANN/dataset/sift100M/D64_N8";
    // partitionDisk_kmeans(file_path, baseFolder);

    // // partitionDataset_kmeans(file_path, baseFolder);
    // // partitionDataset_direct(file_path, baseFolder);

    // nvcc ../partition/partition.cpp ../partition/disk_partition.cpp ../partition/kmeans.cpp ../partition/kmeans.cu ../merge/merge.cpp ../merge/merge.cu ../utils/indexIO.cpp ../utils/datasetIO.cpp ../utils/distance.cpp gpuManagement.cpp scheduler.cpp -I/home/lanlu/raft/cpp/include/ -I/home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include -I/home/lanlu/miniconda3/envs/rapids_raft/include -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx -I/home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include -I/home/lanlu/raft/cpp/build/_deps/benchmark-src/include -lcudart -ldl -lbenchmark -lpthread -lfmt -L/home/lanlu/raft/cpp/build/_deps/benchmark-build/src -Xcompiler -fopenmp -o testMerge
    // // std::string indexFolder = "/home/lanlu/python/raft-ann-bench/src/datasets/sift100M/index21deg-20shards/";
    // // std::string idxMapFolder = "/home/lanlu/scaleGANN/dataset/sift100M/partitions21deg-20shards/";
    std::string baseFolder = "/home/lanlu/scaleGANN/dataset/sift100M/D64_N8/";
    std::string datasetPath = "/home/lanlu/scaleGANN/dataset/sift100M/base.100M.u8bin";
    mergeAllIndexInFolder(baseFolder, datasetPath);
    return 0;
}

