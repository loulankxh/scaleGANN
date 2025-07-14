#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <boost/program_options.hpp>
#include <mkl.h>

#include "../src/utils/fileUtils.h"
#include "../src/search/search.hpp"
#include "../src/search/disk_search.h"
#include "../DiskANN/include/neighbor.h"
#include "../../DiskANN/include/program_options_utils.hpp"

namespace po = boost::program_options;


template <typename T>
void search(std::string data_file, std::string index_file, std::string query_file, std::string truth_file,
            uint32_t k, uint32_t L, bool use_disk){
    std::vector<std::vector<T>> data;
    std::vector<std::vector<uint32_t>> index;
    std::vector<std::vector<T>> query;
    std::vector<std::vector<uint32_t>> groundTruth;
    if(!use_disk) {
        readFile<T>(data_file, data);
        readIndex(index_file, index);
        // printf("rows: %d, Deg: %d\n", index.size(), index[0].size());
    }
    read_query<T>(query_file, query);
    read_groundTruth(truth_file, groundTruth);

    auto s_time = std::chrono::high_resolution_clock::now();
    uint32_t total_visited = 0;
    uint32_t total_distance_cmp = 0;
    long long totalLatency = 0;
    std::vector<std::vector<uint32_t>> result;
    std::vector<std::vector<float>> distances;
    if(use_disk){
        // search_disk<T>(k, L, data_file, index, query, result, distances, &total_visited, &total_distance_cmp, &totalLatency);
        search_disk_scalegann<T>(k, L, data_file, index_file, query, result, distances, &total_visited, &total_distance_cmp, &totalLatency);
    } else
        search<T>(k, L, data, index, query, result, distances, &total_visited, &total_distance_cmp, &totalLatency);

    auto e_time = std::chrono::high_resolution_clock::now();
    auto searchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time);
    uint32_t n_queries = query.size();
    //    auto qps = (1.0 * query_num) / (1.0 * diff.count());
    printf("Search parameters: top-k: %d, search pool size: %d\n", k, L);
    printf("Esimated time for %d queries take %lld ms\n", n_queries, searchDuration.count());

    double recall = get_recall(n_queries, k, result, groundTruth);
    printf("Average recall is %f\n", recall);
    printf("Total visisted points: %d, total compared distances: %d\n", total_visited, total_distance_cmp);
    printf("Avg visisted points: %f, Avg compared distances: %f\n", ((float)total_visited/n_queries), ((float)total_distance_cmp/n_queries));
    printf("Total latency is: %lld ms, Avg latency is %f ms\n",totalLatency, ((float)totalLatency/n_queries));
}



int main(int argc, char **argv){
    std::string data_file, index_file, query_file, truth_file;
    uint32_t k, L, num_threads;
    bool use_disk=false;

    po::options_description desc{
        program_options_utils::make_program_description("search_scaleGANN", "Search ScaleGANN index.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        po::options_description required_configs("Required");
        required_configs.add_options()("data_file", po::value<std::string>(&data_file)->required(),
                                       "Dataset path.");
        required_configs.add_options()("index_file", po::value<std::string>(&index_file)->required(),
                                       "Index path.");
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       "Query file path.");
        required_configs.add_options()("truth_file", po::value<std::string>(&truth_file)->required(),
                                       "Groundtruth file path.");
        required_configs.add_options()("top_k,K", po::value<uint32_t>(&k)->required(),
                                       "Top-k.");
        required_configs.add_options()("itop_k,L", po::value<uint32_t>(&L)->required(),
                                       "Search candidate list size.");
        
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                        program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("use_disk", po::bool_switch()->default_value(false),
                                       "Load vector data from disk when needed when memory is not enough.");                                 

        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (vm["use_disk"].as<bool>())
            use_disk = true;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    omp_set_num_threads(num_threads);
    mkl_set_num_threads(num_threads);
    printf("Using %d threads\n", num_threads);

    uint32_t suffixType = suffixToType(data_file);
    if(suffixType == 0){ // float
        search<float>(data_file, index_file, query_file, truth_file, k, L, use_disk);
    } else if (suffixType == 2) { // uint8_t
        search<uint8_t>(data_file, index_file, query_file, truth_file, k, L, use_disk);
    }
}