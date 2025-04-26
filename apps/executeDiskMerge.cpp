#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <variant>
#include <cassert>
#include <omp.h>
#include <boost/program_options.hpp>
#include <mkl.h>

#include "../src/merge/disk_merge.h"
#include "../../DiskANN/include/program_options_utils.hpp"

namespace po = boost::program_options;


void mergeDisk(std::string base_folder, const std::string merge_index_path,
        const uint64_t nshards, uint32_t merge_degree, uint32_t build_deg,
        const std::string index_name){
    std::string output_index_file = merge_index_path + "/" + index_name;
    scaleGANN_merge(base_folder,
                nshards, merge_degree, build_deg,
                output_index_file,
                index_name);
}

// There can be multiple index of different construction parameters in a folder
void mergeDisk_allIndexInFolder(const std::string baseFolder, const std::string mergeIndexPath, std::string index_name,
                            uint32_t merge_deg, uint32_t build_deg){
    
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

        if (file.is_regular_file() && file.path().filename().string() == index_name) {

            std::string indexPath = file.path().string();
            
            std::vector<std::vector<uint32_t>> mergedIndex;
            mergeDisk(baseFolder, mergeIndexPath, shardNum, merge_deg, build_deg, index_name);

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

int main(int argc, char **argv) {
    std::string base_folder, merge_folder, index_name;
    uint32_t merge_deg, build_deg, num_threads;

    po::options_description desc{
        program_options_utils::make_program_description("scaleGANN_merge_disk_index", "Merge shard indices from disk to get merged index of original vector dataset.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        po::options_description required_configs("Required");
        required_configs.add_options()("base_folder", po::value<std::string>(&base_folder)->required(),
                                       "Folder path where all the partioned data shards are stored.");
        required_configs.add_options()("index_name", po::value<std::string>(&index_name)->required(),
                                       "Name of specific index.");
        required_configs.add_options()("merge_degree,R", po::value<uint32_t>(&merge_deg)->required(),
                                       "Expected degree of the merged index.");
        required_configs.add_options()("build_degree,B", po::value<uint32_t>(&build_deg)->required(),
                                       "Build degree of the each shard index.");


        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                        "Number of threads used.");

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
    printf("Merge degree is %d\n", merge_deg);
    printf("Build degree is %d\n", build_deg);

    merge_folder = base_folder + "/mergedIndex";
    printf("Merge folder is %s\n",merge_folder.c_str());

    mergeDisk_allIndexInFolder(base_folder, merge_folder, index_name, merge_deg, build_deg);
    return 0;
}


