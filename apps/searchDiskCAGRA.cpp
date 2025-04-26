#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <queue>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <stdexcept>
#include <cassert>
#include <chrono>
#include <set>
#include <map>
#include <omp.h>
#include <boost/program_options.hpp>
#include <mkl.h>

#include "../src/utils/fileUtils.h"
#include "../src/search/search.hpp"
#include "../src/search/disk_search.h"
#include "../DiskANN/include/neighbor.h"
#include "../../DiskANN/include/program_options_utils.hpp"

namespace po = boost::program_options;

void mergeResultCAGRA(std::vector<diskann::NeighborPriorityQueue>& mergedResult,
        std::vector<std::vector<uint32_t>>& result,
        std::vector<std::vector<float>>& distances){

    uint32_t n_queries = result.size();
    assert(n_queries == distances.size());

    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0 ; i < n_queries; i ++){
        uint32_t resNum = result[i].size();
        assert(resNum == distances[i].size());

        for (uint32_t j = 0; j < resNum; j++){
            mergedResult[i].insert(diskann::Neighbor((result[i][j]), distances[i][j]));
        }
    }
}


void getResultId(std::vector<diskann::NeighborPriorityQueue>& mergedResult,
        std::vector<std::vector<uint32_t>>& result,
        uint32_t k){
    
    uint32_t n_queries = result.size();
    assert(n_queries == mergedResult.size());

    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0 ; i < n_queries; i ++){
        result[i].resize(k);
        assert(k == mergedResult[i].size());
        for (uint32_t nb_rank = 0; nb_rank < k; nb_rank ++){
            result[i][nb_rank] = mergedResult[i][nb_rank].id;
        }
    }

}


template <typename T>
void searchNaiveCAGRA(std::string data_file, std::string query_file, std::string truth_file, std::string index_dir, std::string index_name,
                    uint32_t k, uint32_t L, uint32_t shard_num){
    std::vector<std::vector<T>> query;
    std::vector<std::vector<uint32_t>> groundTruth;

    read_query<T>(query_file, query);
    read_groundTruth(truth_file, groundTruth);

    printf("finishing loading except index\n");


    uint32_t total_visited = 0;
    uint32_t total_distance_cmp = 0;
    long long totalLatency = 0;

    uint32_t offset = 0;
    long long totalSearchDuration = 0;
    long long totalMergeDuration = 0;
    uint32_t n_queries = query.size();
    std::vector<diskann::NeighborPriorityQueue> mergedResult(n_queries, diskann::NeighborPriorityQueue(k));
    std::vector<std::vector<uint32_t>> final_result(n_queries);
    for (uint32_t iter = 0 ; iter < shard_num; iter++){
        std::string index_file = index_dir + "/partition" + std::to_string(iter) + "/index/" + index_name;

        std::vector<std::vector<uint32_t>> index;
        readIndex(index_file, index);
        printf("Finishing loading index shard %d\n", iter);

        auto s_time = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<uint32_t>> result;
        std::vector<std::vector<float>> distances;
        printf("Starting search\n");

        searchNaiveCAGRA_disk<T>(k, L, offset, data_file, index, query, result, distances, &total_visited, &total_distance_cmp, &totalLatency);
        offset += index.size();

        auto e_time = std::chrono::high_resolution_clock::now();
        auto searchDuration = (std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time)).count();
        printf("Shard %d has search time %lld ms\n", iter, searchDuration);
        totalSearchDuration += searchDuration;


        s_time = std::chrono::high_resolution_clock::now();
        mergeResultCAGRA(mergedResult, result, distances);
        e_time = std::chrono::high_resolution_clock::now();
        auto mergeDuration = (std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time)).count();
        printf("Shard %d has result merge time %lld ms\n", iter, mergeDuration);
        totalMergeDuration += mergeDuration;
    }

    auto s_time = std::chrono::high_resolution_clock::now();
    getResultId(mergedResult, final_result, k);
    auto e_time = std::chrono::high_resolution_clock::now();
    auto fetchIDDuration = (std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time)).count();
    printf("Fetch final result ID takes %lld ms\n", fetchIDDuration);
    totalMergeDuration += fetchIDDuration;

    printf("Total search time %lld ms\n", totalSearchDuration);
    printf("Total merge time %lld ms\n", totalMergeDuration);
    printf("Total time %lld ms\n", totalSearchDuration+totalMergeDuration);


    double recall = get_recall(n_queries, k, final_result, groundTruth);
    printf("Average recall is %f\n", recall);
    printf("Total visisted points: %d, total compared distances: %d\n", total_visited, total_distance_cmp);
    printf("Avg visisted points: %f, Avg compared distances: %f\n", ((float)total_visited/n_queries), ((float)total_distance_cmp/n_queries));
    printf("Total latency is: %lld ms, Avg latency is %f ms\n",totalLatency, ((float)totalLatency/n_queries));
}

template <typename T>
void searchCAGRA(std::string data_file, std::string query_file, std::string truth_file, std::string index_dir, std::string index_name,
                uint32_t k, uint32_t L, uint32_t shard_num){
    std::vector<std::vector<T>> query;
    std::vector<std::vector<uint32_t>> groundTruth;

    read_query<T>(query_file, query);
    read_groundTruth(truth_file, groundTruth);

    printf("finishing loading except index\n");

    
    uint32_t total_visited = 0;
    uint32_t total_distance_cmp = 0;
    long long totalLatency = 0;


    long long totalSearchDuration = 0;
    long long totalMergeDuration = 0;
    uint32_t n_queries = query.size();
    std::vector<diskann::NeighborPriorityQueue> mergedResult(n_queries, diskann::NeighborPriorityQueue(k));
    std::vector<std::vector<uint32_t>> final_result(n_queries);
    for (uint32_t iter = 0 ; iter < shard_num; iter++){
        std::string index_file = index_dir + "/partition" + std::to_string(iter) + "/index/" + index_name;
        std::vector<std::vector<uint32_t>> index;
        readIndex(index_file, index);

        std::string idx_file = index_dir + "/partition" + std::to_string(iter) + "/idmap.ibin";
        uint32_t header[2];
        readMetadata(idx_file, header);
        std::vector<std::vector<uint32_t>> idx_vec(header[0], std::vector<uint32_t>(header[1]));
        readFile<uint32_t>(idx_file, idx_vec); 

        printf("Finishing loading index shard %d\n", iter);


        auto s_time = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<uint32_t>> result;
        std::vector<std::vector<float>> distances;
        printf("Starting search\n");

        searchTranslatedCAGRA<uint8_t>(k, L, data_file, index, idx_vec, query, result, distances, &total_visited, &total_distance_cmp, &totalLatency);

        auto e_time = std::chrono::high_resolution_clock::now();
        auto searchDuration = (std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time)).count();
        printf("Shard %d has search time %lld ms\n", iter, searchDuration);
        totalSearchDuration += searchDuration;


        s_time = std::chrono::high_resolution_clock::now();
        mergeResultCAGRA(mergedResult, result, distances);
        e_time = std::chrono::high_resolution_clock::now();
        auto mergeDuration = (std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time)).count();
        printf("Shard %d has result merge time %lld ms\n", iter, mergeDuration);
        totalMergeDuration += mergeDuration;
    }

    auto s_time = std::chrono::high_resolution_clock::now();
    getResultId(mergedResult, final_result, k);
    auto e_time = std::chrono::high_resolution_clock::now();
    auto fetchIDDuration = (std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time)).count();
    printf("Fetch final result ID takes %lld ms\n", fetchIDDuration);
    totalMergeDuration += fetchIDDuration;

    printf("Total search time %lld ms\n", totalSearchDuration);
    printf("Total merge time %lld ms\n", totalMergeDuration);
    printf("Total time %lld ms\n", totalSearchDuration+totalMergeDuration);



    double recall = get_recall(n_queries, k, final_result, groundTruth);
    printf("Average recall is %f\n", recall);
    printf("Total visisted points: %d, total compared distances: %d\n", total_visited, total_distance_cmp);
    printf("Avg visisted points: %f, Avg compared distances: %f\n", ((float)total_visited/n_queries), ((float)total_distance_cmp/n_queries));
    printf("Total latency is: %lld ms, Avg latency is %f ms\n",totalLatency, ((float)totalLatency/n_queries));
}


int main(int argc, char **argv){
    std::string data_file, query_file, truth_file, index_dir, index_name;
    uint32_t k, L, num_threads;
    uint32_t shard_num;

    po::options_description desc{
        program_options_utils::make_program_description("search_cagra", "Search CAGRA index.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        po::options_description required_configs("Required");
        required_configs.add_options()("data_file", po::value<std::string>(&data_file)->required(),
                                       "Dataset path.");
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       "Query file path.");
        required_configs.add_options()("truth_file", po::value<std::string>(&truth_file)->required(),
                                       "Groundtruth file path.");
        required_configs.add_options()("index_dir", po::value<std::string>(&index_dir)->required(),
                                       "Index folder path where shard inices are stored.");
        required_configs.add_options()("index_name", po::value<std::string>(&index_name)->required(),
                                       "File name of each shard index.");
        required_configs.add_options()("top_k,K", po::value<uint32_t>(&k)->required(),
                                       "Top-k.");
        required_configs.add_options()("L", po::value<uint32_t>(&L)->required(),
                                       "Search candidate list size");
        required_configs.add_options()("shard_num,N", po::value<uint32_t>(&shard_num)->required(),
                                       "Number of shards.");
        
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                        program_options_utils::NUMBER_THREADS_DESCRIPTION);                               

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

    uint32_t suffixType = suffixToType(data_file);
    if(suffixType == 0){ // float
        searchNaiveCAGRA<float>(data_file, query_file, truth_file, index_dir, index_name, k, L, shard_num);
    } else if (suffixType == 2) { // uint8_t
        searchNaiveCAGRA<uint8_t>(data_file, query_file, truth_file, index_dir, index_name, k, L, shard_num);
    }
}