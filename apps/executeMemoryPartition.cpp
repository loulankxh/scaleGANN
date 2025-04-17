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

#include "../src/partition/partition.h"
#include "../src/partition/disk_partition.h"
#include "../src/utils/fileUtils.h"

#define MAX_ITERATION 15
#define SAMPLING_RATE 0.05


void partitionDataset_kmeans(const std::string file_path, std::string baseFolder,
                        uint32_t memory_budget, uint32_t build_deg, uint32_t max_duplication, double epsilon,
                        uint32_t minimum_partition_required, uint32_t max_iteration, double sampling_rate){
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

    std::vector<std::vector<uint32_t>> idx_map(0);
    std::variant<std::vector<std::vector<std::vector<float>>>,
                std::vector<std::vector<std::vector<uint32_t>>>, std::vector<std::vector<std::vector<uint8_t>>>> partitions;
    if(suffixType == 0){ // float
        try {
            auto& floatVec = std::get<std::vector<std::vector<float>>>(vec);
            partitions = main_partitions<float>(memory_budget, npts, ndim, build_deg, max_iteration, max_duplication, minimum_partition_required, floatVec, idx_map);
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    } else if (suffixType == 1) { // uint32_t
        try {
            auto& uint32Vec = std::get<std::vector<std::vector<uint32_t>>>(vec);
            partitions = main_partitions<uint32_t>(memory_budget, npts, ndim, build_deg, max_iteration, max_duplication, minimum_partition_required, uint32Vec, idx_map);
        } catch (const std::bad_variant_access& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    } else if (suffixType == 2) { // uint8_t
        try {
            auto& uint8Vec = std::get<std::vector<std::vector<uint8_t>>>(vec);
            partitions = main_partitions<uint8_t>(memory_budget, npts, ndim, build_deg, max_iteration, max_duplication, minimum_partition_required, uint8Vec, idx_map);
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

        uint32_t partition_lower_bound = 0;
        uint32_t size_limit = 0;
        get_partition_num<float>(memGPU, npts, ndim, degree, 1, &partition_lower_bound, &size_limit);
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

        uint32_t partition_lower_bound = 0;
        uint32_t size_limit = 0;
        get_partition_num<uint32_t>(memGPU, npts, ndim, degree, 1, &partition_lower_bound, &size_limit);
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

        int32_t partition_lower_bound = 0;
        uint32_t size_limit = 0;
        get_partition_num<uint8_t>(memGPU, npts, ndim, degree, 1, &partition_lower_bound, &size_limit);
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

int main() {
    std::string data_path, base_folder;
    uint32_t memory_budget, build_deg, max_duplication, minimum_partition_num, max_iteration;
    double epsilon, sampling_rate;
    uint32_t num_threads;

    po::options_description desc{
        program_options_utils::make_program_description("scaleGANN_partition_disk_dataset", "Partition a vector dataset from disk to get data shards.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        po::options_description required_configs("Required");
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       "File path of the dataset.");
        required_configs.add_options()("base_folder", po::value<std::string>(&base_folder)->required(),
                                       "Folder path where all the partioned data shards are stored.");
        required_configs.add_options()("memory_budget,M", po::value<uint32_t>(&memory_budget)->required(),
                                       "Memory budget for build.");
        required_configs.add_options()("build_degree,R", po::value<uint32_t>(&build_deg)->required(),
                                       "Expected build degree of each shard index.");
        required_configs.add_options()("max_duplication,W", po::value<uint32_t>(&max_duplication)->required(),
                                       "Max duplication time of each vector.");
        required_configs.add_options()("epsilon,E", po::value<double>(&epsilon)->required(),
                                       "Selective duplication factor epsilon.");
        required_configs.add_options()("minimum_partition_num,N", po::value<uint32_t>(&minimum_partition_num)->required(),
                                       "Minimum number of partitioned data shards required by users.");

        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                        "Number of threads used.");
        optional_configs.add_options()("sampling_rate",
                                        po::value<double>(&sampling_rate)->default_value(SAMPLING_RATE),
                                        "Suggested dataset sampling rate for kmeans training.");
        optional_configs.add_options()("max_iteration",
                                        po::value<uint32_t>(&max_iteration)->default_value(MAX_ITERATION),
                                        "Maximum iteration for kmeans training. Default is 15.");

        desc.add(required_configs).add(optional_configs);


        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    omp_set_num_threads(num_threads);
    mkl_set_num_threads(num_threads);
    printf("Using %d threads\n", num_threads);
    printf("Build degree is %d\n", build_deg);

    partitionDataset_kmeans(data_path, base_folder,
                            memory_budget, build_deg, max_duplication, epsilon,
                            minimum_partition_required, max_iteration, sampling_rate);
    // // partitionDataset_direct(file_path, baseFolder);
    return 0;
}

