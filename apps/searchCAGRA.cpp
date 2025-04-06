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

#include "../src/utils/fileUtils.h"
#include "../src/search/search.hpp"
#include "../DiskANN/include/neighbor.h"

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
void testNaiveCAGRA(){
    uint32_t shard_num = 10;

    std::string data_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/base.100M.u8bin";
    std::vector<std::vector<T>> data;
    std::string query_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/query.public.10K.u8bin";
    std::vector<std::vector<T>> query;
    std::string truth_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/groundtruth.neighbors.ibin";
    std::vector<std::vector<uint32_t>> groundTruth;

    readExceptIndex(data_file, data, query_file, query, truth_file, groundTruth);

    printf("finishing loading except index\n");

    
    uint32_t k = 1;
    uint32_t L = 50;
    uint32_t total_visited = 0;
    uint32_t total_distance_cmp = 0;
    long long totalLatency = 0;

    uint32_t offset = 0;
    std::string index_dir = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/indexNaiveShard/";
    long long totalSearchDuration = 0;
    long long totalMergeDuration = 0;
    uint32_t n_queries = query.size();
    std::vector<diskann::NeighborPriorityQueue> mergedResult(n_queries, diskann::NeighborPriorityQueue(k));
    std::vector<std::vector<uint32_t>> final_result(n_queries);
    for (uint32_t iter = 0 ; iter < shard_num; iter++){
        std::string index_file = index_dir + "index" + std::to_string(iter) + "/raft_cagra.graph_degree32.intermediate_graph_degree32.graph_build_algoNN_DESCENT";

        std::vector<std::vector<uint32_t>> index;
        readIndex(index_file, index);
        printf("Finishing loading index shard %d\n", iter);

        auto s_time = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<uint32_t>> result;
        std::vector<std::vector<float>> distances;
        printf("Starting search\n");

        searchNaiveCAGRA<uint8_t>(k, L, offset, data, index, query, result, distances, &total_visited, &total_distance_cmp, &totalLatency);
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
void testCAGRA(){
    uint32_t shard_num = 9;

    std::string data_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/base.100M.u8bin";
    std::vector<std::vector<T>> data;
    std::string query_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/query.public.10K.u8bin";
    std::vector<std::vector<T>> query;
    std::string truth_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/groundtruth.neighbors.ibin";
    std::vector<std::vector<uint32_t>> groundTruth;

    readExceptIndex(data_file, data, query_file, query, truth_file, groundTruth);

    printf("finishing loading except index\n");

    
    uint32_t k = 100;
    uint32_t L = 128;
    uint32_t total_visited = 0;
    uint32_t total_distance_cmp = 0;
    long long totalLatency = 0;


    std::string index_dir = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/indexLessDeg/";
    std::string idxMapFolder = "../../dataset/sift100M/partitionsTest/";
    long long totalSearchDuration = 0;
    long long totalMergeDuration = 0;
    uint32_t n_queries = query.size();
    std::vector<diskann::NeighborPriorityQueue> mergedResult(n_queries, diskann::NeighborPriorityQueue(k));
    std::vector<std::vector<uint32_t>> final_result(n_queries);
    for (uint32_t iter = 0 ; iter < shard_num; iter++){
        std::string index_file = index_dir + "index" + std::to_string(iter) + "/raft_cagra.graph_degree21.intermediate_graph_degree21.graph_build_algoNN_DESCENT";
        std::vector<std::vector<uint32_t>> index;
        readIndex(index_file, index);

        std::string idx_file = idxMapFolder + "idmap" + std::to_string(iter) + ".ibin";
        uint32_t header[2];
        readMetadata(idx_file, header);
        std::vector<std::vector<uint32_t>> idx_vec(header[0], std::vector<uint32_t>(header[1]));
        readFile<uint32_t>(idx_file, idx_vec); 

        printf("Finishing loading index shard %d\n", iter);


        auto s_time = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<uint32_t>> result;
        std::vector<std::vector<float>> distances;
        printf("Starting search\n");

        searchTranslatedCAGRA<uint8_t>(k, L, data, index, idx_vec, query, result, distances, &total_visited, &total_distance_cmp, &totalLatency);

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


// nvcc searchCAGRA.cpp search.cpp ../utils/indexIO.cpp ../utils/datasetIO.cpp ../utils/distance.cpp  -I/home/lanlu/raft/cpp/include/ -I/home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include -I/home/lanlu/miniconda3/envs/rapids_raft/include -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx -I/home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include -I/home/lanlu/raft/cpp/build/_deps/benchmark-src/include -lcudart -ldl -lbenchmark -lpthread -lfmt -L/home/lanlu/raft/cpp/build/_deps/benchmark-build/src -Xcompiler -fopenmp -o testCAGRA
int main(){
    // testCAGRA<uint8_t>();
    testNaiveCAGRA<uint8_t>();
}