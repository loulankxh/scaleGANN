#include <string>
#include <filesystem>
#include <chrono>
#include <boost/program_options.hpp>

#include "../../src/utils/fileUtils.h"
#include "../../DiskANN/include/disk_utils.h"
#include "../../DiskANN/include/index.h"
#include "../../DiskANN/include/program_options_utils.hpp"

namespace po = boost::program_options;


template <typename T>
void build_index(diskann::Metric compareMetric, uint32_t shard_base_dim, uint32_t shard_base_pts,
            const std::shared_ptr<diskann::IndexWriteParameters> low_degree_params, 
            const std::string shard_base_file, const std::string shard_index_file){
    
    diskann::Index<T> _index(compareMetric, shard_base_dim, shard_base_pts,
        low_degree_params, nullptr,
        0, false, false, false, false,
        0, false);

    _index.build(shard_base_file.c_str(), shard_base_pts);

    _index.save(shard_index_file.c_str());
}


int main(int argc, char **argv){
    auto startTime = std::chrono::high_resolution_clock::now();

    std::string base_folder, data_type, dist_fn, data_postfix;
    uint32_t partition_num, R, L;
    uint32_t num_threads, disk_PQ, build_PQ;



    po::options_description desc{
        program_options_utils::make_program_description("build_disk_index", "Build a disk-based index.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("base_folder", po::value<std::string>(&base_folder)->required(),
                                       "Base folder for storing all the partitions.");
        required_configs.add_options()("partition_num,N", po::value<uint32_t>(&partition_num)->default_value(100),
                                       "Number of shards we have.");
        required_configs.add_options()("degree,R", po::value<uint32_t>(&R)->default_value(100),
                                       program_options_utils::MAX_BUILD_DEGREE);
        required_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);

        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("PQ_disk_bytes", po::value<uint32_t>(&disk_PQ)->default_value(0),
                                       "Number of bytes to which vectors should be compressed "
                                       "on SSD; 0 for no compression");
        optional_configs.add_options()("append_reorder_data", po::bool_switch()->default_value(false),
                                       "Include full precision data in the index. Use only in "
                                       "conjuction with compressed data on SSD.");
        optional_configs.add_options()("build_PQ_bytes", po::value<uint32_t>(&build_PQ)->default_value(0),
                                       program_options_utils::BUIlD_GRAPH_PQ_BYTES); // Number of PQ bytes to build the index; 0 for full precision build

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


    if (data_type == "float") data_postfix = ".fbin";
    else if (data_type == "uint8") data_postfix = ".u8bin";
    else if (data_type == "uint32") data_postfix = ".ibin";
    else {
        throw std::invalid_argument("Data type invalid.");
    }

    diskann::Metric metric;
    if (dist_fn == std::string("l2"))
        metric = diskann::Metric::L2;
    else if (dist_fn == std::string("mips"))
        metric = diskann::Metric::INNER_PRODUCT;
    else if (dist_fn == std::string("cosine"))
        metric = diskann::Metric::COSINE;
    else
    {
        std::cout << "Error. Only l2 and mips distance functions are supported" << std::endl;
        return -1;
    }

    // diskann::IndexWriteParameters low_degree_params = diskann::IndexWriteParametersBuilder(L, 2 * R / 3)
    //                                                           .with_filter_list_size(0) // Lf
    //                                                           .with_saturate_graph(false)
    //                                                           .with_num_threads(num_threads)
    //                                                           .build();
    diskann::IndexWriteParameters low_degree_params = diskann::IndexWriteParametersBuilder(L, R)
                                                              .with_filter_list_size(0) // Lf
                                                              .with_saturate_graph(false)
                                                              .with_num_threads(num_threads)
                                                              .build();

    for (uint32_t shard_id = 0; shard_id < partition_num; shard_id++){
        std::string shard_data_path = base_folder + "/partition" + std::to_string(shard_id) + "/data" + data_postfix;
        std::string shard_index_folder = base_folder + "/partition" + std::to_string(shard_id) + "/index/";
        if (!std::filesystem::exists(shard_index_folder)) {
            if (std::filesystem::create_directories(shard_index_folder)) {
                std::cout << "Dir doesn't exists but success to create such dir: " << shard_index_folder << std::endl;
            } else {
                std::cerr << "Dir doesn't exists and fail to create such dir: " << shard_index_folder << std::endl;
            }
        }
        std::string shard_index_path = shard_index_folder + "R" + std::to_string(R) + "_L" + std::to_string(L) + ".ibin";

        uint32_t shard_base_dim, shard_base_pts;
        std::ifstream reader(shard_data_path.c_str(), std::ios::binary);
        reader.read((char *)&shard_base_pts, sizeof(uint32_t));
        reader.read((char *)&shard_base_dim, sizeof(uint32_t));

        if (data_type == "float") {
            build_index<float>(metric, shard_base_dim, shard_base_pts, std::make_shared<diskann::IndexWriteParameters>(low_degree_params),
                                shard_data_path, shard_index_path);
        }
        else if (data_type == "uint8") {
            build_index<uint8_t>(metric, shard_base_dim, shard_base_pts, std::make_shared<diskann::IndexWriteParameters>(low_degree_params),
                                shard_data_path, shard_index_path);
        }
        // else if (data_type == "uint32") {
        //     build_index<uint32_t>(metric, shard_base_dim, shard_base_pts, std::make_shared<diskann::IndexWriteParameters>(low_degree_params),
        //                         shard_data_path, shard_index_path);
        // }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    printf("DiskANN build duration: %lld milliseconds\n", overallDuration.count());
}