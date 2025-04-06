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

        // if(i == 9999) printf("Query %d merge: ");
        for (uint32_t j = 0; j < resNum; j++){
            // if(i == 9999)  printf("%d--%f ", result[i][j]+offset, distances[i][j]);
            mergedResult[i].insert(diskann::Neighbor((result[i][j]+offset), distances[i][j]));
        }
        // if(i == 9999) printf("\n");
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
void testGGNN(){
    uint32_t shard_num = 100;
    uint32_t N_shard = 1000000;
    uint32_t Layer = 4;
    uint32_t SegmentSize = 32;
    uint32_t KBuild = 20;
    std::vector<uint32_t> Ns;
    std::vector<uint32_t> Ns_offsets;
    std::vector<uint32_t> STs_offsets;

    std::string data_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/base.100M.u8bin";
    std::vector<std::vector<T>> data;
    std::string query_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/query.public.10K.u8bin";
    std::vector<std::vector<T>> query;
    std::string truth_file = "/home/lanlu/raft/python/raft-ann-bench/src/datasets/sift100M/groundtruth.neighbors.ibin";
    std::vector<std::vector<uint32_t>> groundTruth;

    readExceptIndex(data_file, data, query_file, query, truth_file, groundTruth);

    printf("finishing loading except index\n");

    
    uint32_t k = 10;
    uint32_t L = 50;
    uint32_t total_visited = 0;
    uint32_t total_distance_cmp = 0;
    long long totalLatency = 0;


    std::string test_dir = "/home/lanlu/ggnn/build_local/";
    long long totalSearchDuration = 0;
    long long totalMergeDuration = 0;
    uint32_t n_queries = query.size();
    std::vector<diskann::NeighborPriorityQueue> mergedResult(n_queries, diskann::NeighborPriorityQueue(k));
    std::vector<std::vector<uint32_t>> final_result(n_queries);
    for (uint32_t iter = 0 ; iter < shard_num; iter++){
        std::string test_file = test_dir + "part_" + std::to_string(iter) + ".ggnn";
        Ns.clear();
        Ns_offsets.clear();
        STs_offsets.clear();

        uint32_t G = 0;
        uint32_t N_all = 0;
        uint32_t ST_all = 0;
        std::vector<std::vector<uint32_t>> index;
        std::vector<uint32_t> translation;
        loadGGNNOneIndex<uint32_t, float>(test_file, N_shard, Layer, SegmentSize, KBuild, G, N_all, ST_all, Ns, Ns_offsets, STs_offsets, index, translation);

        printf("loading index shard %d\n", iter);

        uint32_t offset = iter * N_shard;
        std::vector<std::vector<T>> segmentedData;
        segmentedData.insert(segmentedData.end(), data.begin() + offset, data.begin() + offset + N_shard);
        printf("Data segment %d has size %d\n", iter, segmentedData.size());

        auto s_time = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<uint32_t>> result;
        std::vector<std::vector<float>> distances;
        printf("Starting search\n");

        search<T>(k, L, segmentedData, index, query, result, distances, 
                &total_visited, &total_distance_cmp, &totalLatency, 0, 1000000);
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


// nvcc searchGGNN.cpp search.cpp ../utils/indexIO.cpp ../utils/datasetIO.cpp ../utils/distance.cpp  -I/home/lanlu/raft/cpp/include/ -I/home/lanlu/miniconda3/envs/rapids_raft/targets/x86_64-linux/include -I/home/lanlu/miniconda3/envs/rapids_raft/include -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids -I/home/lanlu/miniconda3/envs/rapids_raft/include/rapids/libcudacxx -I/home/lanlu/raft/cpp/build/_deps/nlohmann_json-src/include -I/home/lanlu/raft/cpp/build/_deps/benchmark-src/include -lcudart -ldl -lbenchmark -lpthread -lfmt -L/home/lanlu/raft/cpp/build/_deps/benchmark-build/src -Xcompiler -fopenmp -o testGGNN
int main(){
    testGGNN<uint8_t>();
}