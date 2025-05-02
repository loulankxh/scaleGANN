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

void mergeResultGGNN(std::vector<diskann::NeighborPriorityQueue>& mergedResult,
        std::vector<std::vector<uint32_t>>& result,
        std::vector<std::vector<float>>& distances,
        uint32_t offset){

    uint32_t n_queries = result.size();
    assert(n_queries == distances.size());

    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0 ; i < n_queries; i ++){
        uint32_t resNum = result[i].size();
        assert(resNum == distances[i].size());

        for (uint32_t j = 0; j < resNum; j++){
            mergedResult[i].insert(diskann::Neighbor((result[i][j]+offset), distances[i][j]));
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
void searchGGNN(std::string data_type, 
                std::string data_dir, std::string query_file, std::string truth_file, std::string index_dir, std::string index_name,
                uint32_t k, uint32_t L,
                uint32_t shard_num, uint32_t N_shard, uint32_t Layer, uint32_t SegmentSize, uint32_t KBuild,
                bool use_disk){
    std::vector<uint32_t> Ns;
    std::vector<uint32_t> Ns_offsets;
    std::vector<uint32_t> STs_offsets;

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
        std::string index_shard_file = index_dir + "/partition" + std::to_string(iter) + "/" + index_name;
        Ns.clear();
        Ns_offsets.clear();
        STs_offsets.clear();

        uint32_t G = 0;
        uint32_t N_all = 0;
        uint32_t ST_all = 0;
        std::vector<std::vector<uint32_t>> index;
        std::vector<uint32_t> translation;
        loadGGNNOneIndex<uint32_t, float>(index_shard_file, N_shard, Layer, SegmentSize, KBuild, G, N_all, ST_all, Ns, Ns_offsets, STs_offsets, index, translation);

        printf("loading index shard %d\n", iter);

        uint32_t offset = iter * N_shard;
        std::string data_shard_file = data_dir + "/partition" + std::to_string(iter) + "/data";
        // if(data_type == "float"){
        //     data_shard_file = data_shard_file + ".fvecs";
        // } else if(data_type == "uint8"){
        //     data_shard_file = data_shard_file + ".bvecs";
        // }
        if(data_type == "float"){
            data_shard_file = data_shard_file + ".fbin";
        } else if(data_type == "uint8"){
            data_shard_file = data_shard_file + ".u8bin";
        }
        std::vector<std::vector<T>> segmentedData;
        if(!use_disk){
            readFile<T>(data_shard_file, segmentedData);
            printf("Data segment %d has size %d\n", iter, segmentedData.size());
        }

        auto s_time = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<uint32_t>> result;
        std::vector<std::vector<float>> distances;
        printf("Starting search\n");

        if(use_disk)
            search_disk<T>(k, L, data_shard_file, index, query, result, distances, 
                &total_visited, &total_distance_cmp, &totalLatency, N_shard, 0, N_shard);
        else
            search<T>(k, L, segmentedData, index, query, result, distances, 
                &total_visited, &total_distance_cmp, &totalLatency, 0, N_shard);
        // hierarchicalSearch<T>(N_all, ST_all, Ns, Ns_offsets, STs_offsets, translation,
        //             k, L, segmentedData, index, query, result, distances,
        //             &total_visited, &total_distance_cmp, &totalLatency);

        auto e_time = std::chrono::high_resolution_clock::now();
        auto searchDuration = (std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time)).count();
        printf("Shard %d has search time %lld ms\n", iter, searchDuration);
        totalSearchDuration += searchDuration;


        s_time = std::chrono::high_resolution_clock::now();
        mergeResultGGNN(mergedResult, result, distances, offset);
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
    // std::string index_dir = "/home/lanlu/ggnn/build_local/";
    std::string data_type, data_dir, query_file, truth_file, index_dir, index_name;
    uint32_t k, L, num_threads;
    // uint32_t shard_num = 100, uint32_t N_shard = 1000000, uint32_t KBuild = 20;
    uint32_t shard_num, N_shard, Layer, SegmentSize, KBuild;
    bool use_disk=false;

    po::options_description desc{
        program_options_utils::make_program_description("search_ggnn", "Search GGNN index.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       "Type of data vectors.");
        required_configs.add_options()("data_dir", po::value<std::string>(&data_dir)->required(),
                                       "Index folder path where data shards are stored.");
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
        required_configs.add_options()("itop_k,L", po::value<uint32_t>(&L)->required(),
                                       "Search candidate list size.");
        required_configs.add_options()("shard_num,N", po::value<uint32_t>(&shard_num)->required(),
                                       "Number of shards.");
        required_configs.add_options()("shard_size,S", po::value<uint32_t>(&N_shard)->required(),
                                       "Number of points in each shard.");
        required_configs.add_options()("KBuild,R", po::value<uint32_t>(&KBuild)->required(),
                                       "Build degree of the graph index.");   
        
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                        program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("Layer", po::value<uint32_t>(&Layer)->default_value(4),
                                        "Number layers in the index.");
        optional_configs.add_options()("segment_size", po::value<uint32_t>(&SegmentSize)->default_value(32),
                                       "Build segment size.");
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

    if(data_type == "float"){ // float
        searchGGNN<float>(data_type, data_dir, query_file, truth_file, index_dir, index_name, k, L,
                    shard_num, N_shard, Layer, SegmentSize, KBuild, use_disk);
    } else if (data_type == "uint8") { // uint8_t
        searchGGNN<uint8_t>(data_type, data_dir, query_file, truth_file, index_dir, index_name, k, L,
                    shard_num, N_shard, Layer, SegmentSize, KBuild, use_disk);
    }
}