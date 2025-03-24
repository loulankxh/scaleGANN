#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <random>
#include <stdexcept>
#include <omp.h>
#include <cassert>

#include "../utils/fileUtils.h"
#include "merge.cuh"


void contactAndSort(
    const std::vector<std::vector<std::vector<uint32_t>>>& partitions,
    std::vector<std::vector<uint32_t>>& merged) {
    merged.clear();
    for (const auto& partition : partitions) {
        merged.insert(merged.end(), partition.begin(), partition.end());
    }
    sort(merged.begin(), merged.end(), [](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
        return a[0] < b[0]; 
    });
}


void mergeByFirstDimension(
    const std::vector<std::vector<uint32_t>>& sortedVec, 
    std::vector<std::vector<uint32_t>>& merged) {
    if (sortedVec.empty()) return;

    uint32_t currentKey = sortedVec[0][0];
    std::vector<uint32_t> values;

    for (const auto& row : sortedVec) {
        if (row[0] == currentKey) {
            values.push_back(row[1]);
        } else {
            merged.push_back({currentKey});
            merged.back().insert(merged.back().end(), values.begin(), values.end());
            
            currentKey = row[0];
            values.clear();
            values.push_back(row[1]);
        }
    }
    merged.push_back({currentKey});
    merged.back().insert(merged.back().end(), values.begin(), values.end());
}


void select_random_neighbors(std::vector<std::vector<uint32_t>>& grouped, size_t num_values) {
    // std::random_device rd; 
    // std::mt19937 gen(rd());

    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0 ; i < grouped.size(); i++){
        std::random_device rd; 
        std::mt19937 gen(rd());
        
        auto& group = grouped[i];
        if (group.size() > num_values) { 
            std::vector<uint32_t> values(group.begin(), group.end());
            if (values.size() > num_values) {
                std::shuffle(values.begin(), values.end(), gen);
                values.resize(num_values);
            }
            group.resize(0);
            group.insert(group.end(), values.begin(), values.end());
        }
    }
}


void merge(
    std::vector<std::vector<std::vector<uint32_t>>>& partitions, 
    std::vector<std::vector<uint32_t>>& merged){
    std::vector<std::vector<uint32_t>> mergeSorted;
    contactAndSort(partitions, mergeSorted);
    mergeByFirstDimension(mergeSorted, merged);
    uint32_t deg = (partitions[0].size() > 0) ? partitions[0][0].size() : 0;
    select_random_neighbors(merged, deg);
}


void mergeShardAfterTranslation(omp_lock_t* locks, std::vector<std::vector<uint32_t>>& merged_index,
        std::vector<std::vector<uint32_t>>& index, std::vector<uint32_t>& idx_vec){
    
    uint32_t dataset_size = merged_index.size();
    uint32_t shard_size = index.size();
    assert(shard_size == idx_vec.size());

    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < shard_size; ++i) {
        uint32_t global_index = idx_vec[i];

        auto& neighbor_list = index[i];

        std::vector<uint32_t> global_neighbor_list;
        for (uint32_t neighbor_local_index : neighbor_list) {
            if (neighbor_local_index >= shard_size) {
                throw std::runtime_error("Invalid neighbor local index in neighbor list.");
            }
            global_neighbor_list.push_back(idx_vec[neighbor_local_index]);
        }

        omp_set_lock(&locks[global_index]);
        merged_index[global_index].insert(merged_index[global_index].end(), global_neighbor_list.begin(), global_neighbor_list.end());
        omp_unset_lock(&locks[global_index]);    
    }

}


void mergeShardAfterTranslationGPU(omp_lock_t* locks, std::vector<std::vector<uint32_t>>& merged_index,
            std::vector<std::vector<uint32_t>>& index,
            std::vector<uint32_t>& idx_vec,
            uint32_t gpu_id){
    
    std::vector<std::vector<uint32_t>> translated_index;
    translateShardGPU(translated_index, index, idx_vec, gpu_id);

     uint32_t shard_size = translated_index.size();
     assert(shard_size == index.size());
     assert(shard_size == idx_vec.size());

     #pragma omp parallel for schedule(static)
     for(uint32_t i = 0 ; i < shard_size; i++){
        uint32_t id = translated_index[i][0];

        omp_set_lock(&locks[id]);
        merged_index[id].insert(merged_index[id].end(), translated_index[i].begin() + 1, translated_index[i].end());
        omp_unset_lock(&locks[id]);  

     }
}







void standardizeNeighborList(std::vector<std::vector<uint32_t>>& merged_index, uint32_t deg){
    select_random_neighbors(merged_index, deg);
}

