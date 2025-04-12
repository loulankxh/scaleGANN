#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include "../src/utils/fileUtils.h"
#include "../src/search/search.hpp"
#include "../DiskANN/include/neighbor.h"


void readIndex_DiskANN(std::string index_file, std::vector<std::vector<uint32_t>>& index){
    std::ifstream reader(index_file.c_str(), std::ios::binary);

    size_t expected_file_size;
    reader.read((char *)&expected_file_size, sizeof(uint64_t));
    uint32_t input_width;
    reader.read((char *)&input_width, sizeof(uint32_t));
    uint64_t vamana_index_frozen = 0;
    reader.read((char *)&vamana_index_frozen, sizeof(uint64_t));
    uint32_t medoid;
    reader.read((char *)&medoid, sizeof(uint32_t));

    size_t read_data_size =
        sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t);
    index.resize(0);
    uint32_t nnbrs = 0;
    while (read_data_size < expected_file_size){
        reader.read((char *)&nnbrs, sizeof(uint32_t));
        std::vector<uint32_t> shard_nhood(nnbrs);
        reader.read((char *)shard_nhood.data(), nnbrs * sizeof(uint32_t));
        index.emplace_back(shard_nhood);

        read_data_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
    }
}


template <typename T>
void search(){
    std::string data_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/base.100M.u8bin";
    std::vector<std::vector<T>> data;
    std::string index_file = "/home/lanlu/scaleGANN/dataset/sift100M/DiskANN/mergedIndex/R32_L64.ibin";
    std::vector<std::vector<uint32_t>> index;
    std::string query_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/query.public.10K.u8bin";
    std::vector<std::vector<T>> query;
    std::string truth_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/groundtruth.neighbors.ibin";
    std::vector<std::vector<uint32_t>> groundTruth;
    readExceptIndex<T>(data_file, data, query_file, query, truth_file, groundTruth);
    readIndex_DiskANN(index_file, index);


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



int main(){
    search<uint8_t>();
}