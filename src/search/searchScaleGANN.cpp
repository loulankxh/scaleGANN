#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include "search.hpp"
#include "priorityList.hpp"


template <typename T>
void testOurDesign(){
    std::string data_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/base.100M.u8bin";
    std::vector<std::vector<T>> data;
    std::string index_file = "/home/lanlu/scaleGANN/dataset/sift100M/D64_N8/mergedIndex/raft_cagra.graph_degree32.intermediate_graph_degree32.graph_build_algoNN_DESCENT";
    std::vector<std::vector<uint32_t>> index;
    std::string query_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/query.public.10K.u8bin";
    std::vector<std::vector<T>> query;
    std::string truth_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/groundtruth.neighbors.ibin";
    std::vector<std::vector<uint32_t>> groundTruth;
    read(data_file, data, index_file, index, query_file, query, truth_file, groundTruth);


    auto s_time = std::chrono::high_resolution_clock::now();
    uint32_t k = 10;
    uint32_t L = 128;
    uint32_t total_visited = 0;
    uint32_t total_distance_cmp = 0;
    long long totalLatency = 0;
    std::vector<std::vector<uint32_t>> result;
    std::vector<std::vector<float>> distances;
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



// nvcc searchScaleGANN.cpp search.cpp ../utils/indexIO.cpp ../utils/datasetIO.cpp ../utils/distance.cpp  -I/home/lanlu/raft/cpp/include/ -I/home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include -I/home/lanlu/miniconda3/envs/rapids_raft/include -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx -I/home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include -I/home/lanlu/raft/cpp/build/_deps/benchmark-src/include -lcudart -ldl -lbenchmark -lpthread -lfmt -L/home/lanlu/raft/cpp/build/_deps/benchmark-build/src -Xcompiler -fopenmp -o testOurDesign
int main(){
    testOurDesign<uint8_t>();
}