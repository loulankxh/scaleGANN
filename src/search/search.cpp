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

#include "../utils/fileUtils.h"
#include "../utils/distance.h"
#include "../../DiskANN/include/neighbor.h"

void random_start_points(uint32_t s, uint32_t s_id, uint32_t e_id,
    std::vector<uint32_t>& start){
    
    if(s > (e_id - s_id)){
        s = e_id - s_id;
        printf("Number of selected start points should be no greater than # of points in dataset. Return the entire dataset as start choices ...\n");
        //throw std::invalid_argument("Number of selected start points should be no greater than # of points in dataset.");
        start.clear();
        start.resize(s);
        for (uint32_t count = 0; count < s; count ++){
            start[count] = s_id + count;
        }
        return;
    }

    std::unordered_set<uint32_t> unique_numbers;

    std::random_device rd;  
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(s_id, e_id - 1);

    while (unique_numbers.size() < s) {
        uint32_t num = dist(gen);
        unique_numbers.insert(num);
    }

    start.clear();
    start.insert(start.end(), unique_numbers.begin(), unique_numbers.end());
}


template <typename T>
void search(const uint32_t k, const uint32_t L,
    std::vector<std::vector<T>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<T>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t startLine = 0,
    uint32_t stopLine = 0){

    if (k > L){
        throw std::invalid_argument("L should be no smaller than K.");
    }


    uint32_t npts = data.size();
    uint32_t num_queries = query.size();
    result.resize(num_queries);
    distances.resize(num_queries);

    std::vector<uint32_t> count_visited(num_queries);
    std::vector<uint32_t> count_distances(num_queries);


    #pragma omp parallel for schedule(static)
    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        auto startTime = std::chrono::high_resolution_clock::now();

        result[num_q].resize(k);
        distances[num_q].resize(k);
        std::vector<T> cur_query = query[num_q];
        diskann::NeighborPriorityQueue L_list(L);
        std::vector<diskann::Neighbor> expanded_nodes(0);
        std::unordered_map<uint32_t, bool> visited;


        uint32_t sid = startLine;
        uint32_t eid = (stopLine > 0) ? stopLine: npts;
        std::vector<uint32_t> start(L);
        random_start_points(L, sid, eid, start);
        for (uint32_t id: start){
            std::vector<T> node = data[id];
            float dist = l2_distance_square<T>(cur_query, node);
            count_distances[num_q] ++;
            diskann::Neighbor nn = diskann::Neighbor(id, dist);
            L_list.insert(nn);
        }

        while (L_list.has_unexpanded_node())
        {
            auto nbr = L_list.closest_unexpanded();
            auto n = nbr.id;

            expanded_nodes.emplace_back(nbr);

            std::vector<uint32_t> neighbors = index[n];
            // #pragma omp parallel for schedule(static)
            for (uint32_t iter = 0; iter< neighbors.size(); iter++){
                uint32_t nb_id = neighbors[iter];
                
                assert(nb_id < npts);

                if(visited.count(nb_id) == 0 || ((visited.count(nb_id) > 0) && (visited[nb_id]==false))){
                    visited[nb_id] = true;
                    count_visited[num_q] ++;

                    std::vector<T> node = data[nb_id];
                    float dist = l2_distance_square<T>(cur_query, node);
                    count_distances[num_q] ++;
                    
                    L_list.insert(diskann::Neighbor(nb_id, dist));
                }

            }
        }

        uint32_t pos = 0;
        for (uint32_t i = 0; i < L_list.size(); ++i)
        {
            if (L_list[i].id < npts)
            {   
                result[num_q][pos] = L_list[i].id;
                distances[num_q][pos] = L_list[i].distance;
                pos++;
            }
            if (pos == k)
                break;
        }
        if (pos < k)
        {
            throw std::invalid_argument("Founded nearest neighbors fewer than K elements for current query");
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        (*total_latency) += latency.count();
    }


    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        (*total_visited) += count_visited[num_q];
        (*total_distance_cmp) += count_distances[num_q];
    }

}



template <typename T>
void searchFromLastLayer(const uint32_t k, const uint32_t L,
    std::vector<std::vector<uint32_t>>& startNodes,
    std::vector<std::vector<T>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<uint32_t>& translation,
    std::vector<std::vector<T>>& query,
    std::vector<diskann::NeighborPriorityQueue>& intermediate_result,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency){

    if (k > L){
        throw std::invalid_argument("L should be no smaller than K.");
    }


    uint32_t npts = data.size();
    uint32_t num_queries = query.size();
    intermediate_result.resize(num_queries);

    std::vector<uint32_t> count_visited(num_queries);
    std::vector<uint32_t> count_distances(num_queries);

    #pragma omp parallel for schedule(static)
    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        auto startTime = std::chrono::high_resolution_clock::now();

        std::vector<T> cur_query = query[num_q];
        diskann::NeighborPriorityQueue L_list(L);
        std::vector<diskann::Neighbor> expanded_nodes(0);
        std::unordered_map<uint32_t, bool> visited;

        std::vector<uint32_t> start = startNodes[num_q];
        for (uint32_t id: start){
            std::vector<T> node = data[translation[id]];
            float dist = l2_distance_square<T>(cur_query, node);
            count_distances[num_q] ++;
            diskann::Neighbor nn = diskann::Neighbor(id, dist);
            L_list.insert(nn);
        }

        while (L_list.has_unexpanded_node())
        {
            auto nbr = L_list.closest_unexpanded();
            auto n = nbr.id;

            expanded_nodes.emplace_back(nbr);

            std::vector<uint32_t> neighbors = index[n];
            // #pragma omp parallel for schedule(static)
            for (uint32_t iter = 0; iter< neighbors.size(); iter++){
                uint32_t nb_id = neighbors[iter];
                assert(translation[nb_id] < npts);

                if(visited.count(nb_id) == 0 || ((visited.count(nb_id) > 0) && (visited[nb_id]==false))){
                    visited[nb_id] = true;
                    count_visited[num_q] ++;

                    std::vector<T> node = data[translation[nb_id]];
                    float dist = l2_distance_square<T>(cur_query, node);
                    count_distances[num_q] ++;
                    
                    L_list.insert(diskann::Neighbor(nb_id, dist));
                }

            }
        }


        uint32_t listSize = L_list.size();
        for(uint32_t iter = 0; iter < listSize; iter ++){
            uint32_t id = L_list[iter].id;
            L_list[iter].id = translation[id];
        }
        intermediate_result[num_q] = L_list;


        auto endTime = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        (*total_latency) += latency.count();

    }

    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        (*total_visited) += count_visited[num_q];
        (*total_distance_cmp) += count_distances[num_q];
    }

}


template <typename T>
void searchBottomLayer(const uint32_t k, const uint32_t L,
    std::vector<std::vector<uint32_t>>& startNodes,
    std::vector<std::vector<T>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<T>>& query,
    std::vector<diskann::NeighborPriorityQueue>& intermediate_result,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency){

    if (k > L){
        throw std::invalid_argument("L should be no smaller than K.");
    }


    uint32_t npts = data.size();
    uint32_t num_queries = query.size();
    intermediate_result.resize(num_queries);

    std::vector<uint32_t> count_visited(num_queries);
    std::vector<uint32_t> count_distances(num_queries);

    #pragma omp parallel for schedule(static)
    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        auto startTime = std::chrono::high_resolution_clock::now();

        std::vector<T> cur_query = query[num_q];
        diskann::NeighborPriorityQueue L_list(L);
        std::vector<diskann::Neighbor> expanded_nodes(0);
        std::unordered_map<uint32_t, bool> visited;

        std::vector<uint32_t> start = startNodes[num_q];
        for (uint32_t id: start){
            std::vector<T> node = data[id];
            float dist = l2_distance_square<T>(cur_query, node);
            count_distances[num_q] ++;
            diskann::Neighbor nn = diskann::Neighbor(id, dist);
            L_list.insert(nn);
        }

        while (L_list.has_unexpanded_node())
        {
            auto nbr = L_list.closest_unexpanded();
            auto n = nbr.id;

            expanded_nodes.emplace_back(nbr);

            std::vector<uint32_t> neighbors = index[n];
            // #pragma omp parallel for schedule(static)
            for (uint32_t iter = 0; iter< neighbors.size(); iter++){
                uint32_t nb_id = neighbors[iter];
                assert(nb_id < npts);

                if(visited.count(nb_id) == 0 || ((visited.count(nb_id) > 0) && (visited[nb_id]==false))){
                    visited[nb_id] = true;
                    count_visited[num_q] ++;

                    std::vector<T> node = data[nb_id];
                    float dist = l2_distance_square<T>(cur_query, node);
                    count_distances[num_q] ++;
                    
                    L_list.insert(diskann::Neighbor(nb_id, dist));
                }

            }
        }
        intermediate_result[num_q] = L_list;


        auto endTime = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        (*total_latency) += latency.count();
    }

    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        (*total_visited) += count_visited[num_q];
        (*total_distance_cmp) += count_distances[num_q];
    }
}



template <typename T>
void hierarchicalSearch(uint32_t N_all, uint32_t ST_all,
    std::vector<uint32_t>& Ns,
    std::vector<uint32_t>& Ns_offsets,
    std::vector<uint32_t>& STs_offsets, 
    std::vector<uint32_t>& translation,
    const uint32_t k, const uint32_t L,
    std::vector<std::vector<T>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<T>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency){

    uint32_t layer_num = Ns.size();
    uint32_t num_queries = query.size();

    std::vector<std::vector<uint32_t>> startNodes(num_queries);
    std::vector<diskann::NeighborPriorityQueue> intermediate_result;

    std::vector<uint32_t> count_visited(num_queries);
    std::vector<uint32_t> count_distances(num_queries);

    for(int layer_id = layer_num - 1; layer_id >= 0; layer_id--){
        printf("Prepare searching layer %d\n", layer_id);

        uint32_t current_node_num = Ns[layer_id];
        uint32_t current_node_offset = Ns_offsets[layer_id];
        uint32_t current_translation_offset = STs_offsets[layer_id];

        std::vector<std::vector<uint32_t>> segmentedIndex;
        if (layer_id > 0) segmentedIndex.insert(segmentedIndex.end(), index.begin() + current_node_offset, index.begin() + current_node_offset + current_node_num);

        std::vector<uint32_t> segmentedTranslation;
        if (layer_id > 0) segmentedTranslation.insert(segmentedTranslation.end(), translation.begin() + current_translation_offset, translation.begin() + current_translation_offset + current_node_num);


        // prepare startNodes
        if (layer_id == layer_num - 1){
            std::vector<uint32_t> general_start;
            random_start_points(L, 0, current_node_num, general_start);

            #pragma omp parallel for schedule(static)
            for(uint32_t num_q = 0; num_q < num_queries; num_q ++){
                startNodes[num_q].insert(startNodes[num_q].end(), general_start.begin(), general_start.end());
            }

        } else {
            uint32_t nextStartNodeNum = (intermediate_result[0].size() > L) ? L : intermediate_result[0].size();
            startNodes.clear();
            startNodes.resize(num_queries);

            std::map<uint32_t, uint32_t> inverseTranslationMap; 
            if (layer_id > 0) {
                for (size_t i = 0; i < segmentedTranslation.size(); ++i) {
                    inverseTranslationMap[segmentedTranslation[i]] = i;
                }
            }

            #pragma omp parallel for schedule(static)
            for(uint32_t num_q = 0; num_q < num_queries; num_q ++){
                startNodes[num_q].resize(nextStartNodeNum);
                diskann::NeighborPriorityQueue inter_q = intermediate_result[num_q];
                assert (inter_q.size() >= nextStartNodeNum);
            
                for(uint32_t iter = 0; iter < nextStartNodeNum; iter ++){
                    uint32_t id = inter_q[iter].id;

                    if (layer_id > 0){
                        assert(inverseTranslationMap.find(id) != inverseTranslationMap.end());
                        id = inverseTranslationMap[id];
                    }

                    startNodes[num_q][iter] = id;
                }
            }
    
        }

        // printf("Start searching layer %d\n", layer_id);

        intermediate_result.clear();
        if (layer_id > 0){
            searchFromLastLayer(k, L, startNodes, data, segmentedIndex, segmentedTranslation, query, intermediate_result, total_visited, total_distance_cmp, total_latency); 
        }  else {
            searchBottomLayer(k, L, startNodes, data, index, query, intermediate_result, total_visited, total_distance_cmp, total_latency); 
        }

        // printf("Finish searching layer %d\n", layer_id);


    }


    uint32_t npts = data.size();

    result.resize(num_queries);
    distances.resize(num_queries);

    // printf("Num queries is: %d\n", num_queries);
    #pragma omp parallel for schedule(static)
    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        result[num_q].resize(k);
        distances[num_q].resize(k);

        uint32_t pos = 0;
        diskann::NeighborPriorityQueue inter_q = intermediate_result[num_q];
        // printf("intermediate size: %d\n", inter_q.size());
        for (uint32_t i = 0; i < inter_q.size(); ++i)
        {   
            if (inter_q[i].id < npts)
            {   
                result[num_q][pos] = inter_q[i].id;
                distances[num_q][pos] = inter_q[i].distance;
                pos++;
            }
            if (pos == k)
                break;
        }
        // Lan (To do): Catch error in pragma
        // if (pos < k)
        // {
        //     throw std::invalid_argument("Founded nearest neighbors fewer than K elements for current query");
        // }
    }


}



template <typename T>
void searchNaiveCAGRA(const uint32_t k, const uint32_t L, uint32_t offset,
    std::vector<std::vector<T>>& data,
    std::vector<std::vector<uint32_t>>& index,
    std::vector<std::vector<T>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency){

    if (k > L){
        throw std::invalid_argument("L should be no smaller than K.");
    }


    uint32_t npts = data.size();
    uint32_t num_queries = query.size();
    result.resize(num_queries);
    distances.resize(num_queries);

    std::vector<uint32_t> count_visited(num_queries);
    std::vector<uint32_t> count_distances(num_queries);

    #pragma omp parallel for schedule(static)
    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        auto startTime = std::chrono::high_resolution_clock::now();

        result[num_q].resize(k);
        distances[num_q].resize(k);

        std::vector<T> cur_query = query[num_q];
        diskann::NeighborPriorityQueue L_list(L);
        std::vector<diskann::Neighbor> expanded_nodes(0);
        std::unordered_map<uint32_t, bool> visited;

        std::vector<uint32_t> start(L);
        random_start_points(L, 0, index.size(), start);
        for (uint32_t id: start){
            std::vector<T> node = data[id + offset];
            float dist = l2_distance_square<T>(cur_query, node);
            count_distances[num_q] ++;
            diskann::Neighbor nn = diskann::Neighbor(id, dist);
            L_list.insert(nn);
        }

        while (L_list.has_unexpanded_node())
        {
            auto nbr = L_list.closest_unexpanded();
            auto n = nbr.id;

            expanded_nodes.emplace_back(nbr);

            std::vector<uint32_t> neighbors = index[n];
            // #pragma omp parallel for schedule(static)
            for (uint32_t iter = 0; iter< neighbors.size(); iter++){
                uint32_t nb_id = neighbors[iter];
                assert((nb_id + offset) < npts);

                if(visited.count(nb_id) == 0 || ((visited.count(nb_id) > 0) && (visited[nb_id]==false))){
                    visited[nb_id] = true;
                    count_visited[num_q] ++;

                    std::vector<T> node = data[nb_id + offset];
                    float dist = l2_distance_square<T>(cur_query, node);
                    count_distances[num_q] ++;
                    
                    L_list.insert(diskann::Neighbor(nb_id, dist));
                }

            }
        }


        uint32_t listSize = L_list.size();
        uint32_t pos = 0;
        for (uint32_t iter = 0; iter < listSize; ++iter)
        {   
            uint32_t id = L_list[iter].id;
            id = id + offset;
            if (id < npts)
            {   
                result[num_q][pos] = id;
                distances[num_q][pos] = L_list[iter].distance;
                pos++;
            }
            if (pos == k)
                break;
        }
        if (pos < k)
        {
            throw std::invalid_argument("Founded nearest neighbors fewer than K elements for current query");
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        (*total_latency) += latency.count();
    }

    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        (*total_visited) += count_visited[num_q];
        (*total_distance_cmp) += count_distances[num_q];
    }
}



template <typename T>
void searchTranslatedCAGRA(const uint32_t k, const uint32_t L,
    std::vector<std::vector<T>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint32_t>>& translation,
    std::vector<std::vector<T>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency){

    if (k > L){
        throw std::invalid_argument("L should be no smaller than K.");
    }


    uint32_t npts = data.size();
    uint32_t num_queries = query.size();
    result.resize(num_queries);
    distances.resize(num_queries);

    std::vector<uint32_t> count_visited(num_queries);
    std::vector<uint32_t> count_distances(num_queries);

    #pragma omp parallel for schedule(static)
    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        auto startTime = std::chrono::high_resolution_clock::now();

        result[num_q].resize(k);
        distances[num_q].resize(k);

        std::vector<T> cur_query = query[num_q];
        diskann::NeighborPriorityQueue L_list(L);
        std::vector<diskann::Neighbor> expanded_nodes(0);
        std::unordered_map<uint32_t, bool> visited;

        std::vector<uint32_t> start(L);
        random_start_points(L, 0, index.size(), start);
        for (uint32_t id: start){
            std::vector<T> node = data[translation[id][1]];
            float dist = l2_distance_square<T>(cur_query, node);
            count_distances[num_q] ++;
            diskann::Neighbor nn = diskann::Neighbor(id, dist);
            L_list.insert(nn);
        }

        while (L_list.has_unexpanded_node())
        {
            auto nbr = L_list.closest_unexpanded();
            auto n = nbr.id;

            expanded_nodes.emplace_back(nbr);

            std::vector<uint32_t> neighbors = index[n];
            // #pragma omp parallel for schedule(static)
            for (uint32_t iter = 0; iter< neighbors.size(); iter++){
                uint32_t nb_id = neighbors[iter];
                assert(translation[nb_id][1] < npts);

                if(visited.count(nb_id) == 0 || ((visited.count(nb_id) > 0) && (visited[nb_id]==false))){
                    visited[nb_id] = true;
                    count_visited[num_q] ++;

                    std::vector<T> node = data[translation[nb_id][1]];
                    float dist = l2_distance_square<T>(cur_query, node);
                    count_distances[num_q] ++;
                    
                    L_list.insert(diskann::Neighbor(nb_id, dist));
                }

            }
        }


        uint32_t listSize = L_list.size();
        uint32_t pos = 0;
        for (uint32_t iter = 0; iter < listSize; ++iter)
        {   
            uint32_t id = L_list[iter].id;
            id = translation[id][1];
            if (id < npts)
            {   
                result[num_q][pos] = id;
                distances[num_q][pos] = L_list[iter].distance;
                pos++;
            }
            if (pos == k)
                break;
        }
        if (pos < k)
        {
            throw std::invalid_argument("Founded nearest neighbors fewer than K elements for current query");
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        (*total_latency) += latency.count();
    }

    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        (*total_visited) += count_visited[num_q];
        (*total_distance_cmp) += count_distances[num_q];
    }
}






double get_recall(uint32_t num_queries, uint32_t k,
        std::vector<std::vector<uint32_t>>& result,
        std::vector<std::vector<uint32_t>>& groundTruth){
    double total_recall = 0;
    std::set<uint32_t> gt, res;

    for (size_t i = 0; i < num_queries; i++)
    {
        gt.clear();
        res.clear();

        assert(k <= groundTruth[i].size());
        gt.insert(groundTruth[i].begin(), groundTruth[i].begin() + k);
        res.insert(result[i].begin(), result[i].end());
        uint32_t cur_recall = 0;

        for (auto &v : gt)
        {   
            if (res.find(v) != res.end())
            {
                cur_recall++;
            }
        }

        if (gt.size() != 0){
            total_recall += ((100.0 * cur_recall) / gt.size());
        }    
        else
            total_recall += 100;
    }

    printf("Total recall is %f, num queries is %d\n", total_recall, num_queries);

    return total_recall / (num_queries);

}



template void search<float>(const uint32_t k, const uint32_t L,
    std::vector<std::vector<float>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<float>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t startLine = 0,
    uint32_t stopLine = 0);
template void search<uint32_t>(const uint32_t k, const uint32_t L,
    std::vector<std::vector<uint32_t>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint32_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t startLine = 0,
    uint32_t stopLine = 0);
template void search<uint8_t>(const uint32_t k, const uint32_t L,
    std::vector<std::vector<uint8_t>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint8_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t startLine = 0,
    uint32_t stopLine = 0);


template void hierarchicalSearch<float>(uint32_t N_all, uint32_t ST_all,
    std::vector<uint32_t>& Ns,
    std::vector<uint32_t>& Ns_offsets,
    std::vector<uint32_t>& STs_offsets, 
    std::vector<uint32_t>& translation,
    const uint32_t k, const uint32_t L,
    std::vector<std::vector<float>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<float>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);
template void hierarchicalSearch<uint32_t>(uint32_t N_all, uint32_t ST_all,
    std::vector<uint32_t>& Ns,
    std::vector<uint32_t>& Ns_offsets,
    std::vector<uint32_t>& STs_offsets, 
    std::vector<uint32_t>& translation,
    const uint32_t k, const uint32_t L,
    std::vector<std::vector<uint32_t>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint32_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);
template void hierarchicalSearch<uint8_t>(uint32_t N_all, uint32_t ST_all,
    std::vector<uint32_t>& Ns,
    std::vector<uint32_t>& Ns_offsets,
    std::vector<uint32_t>& STs_offsets, 
    std::vector<uint32_t>& translation,
    const uint32_t k, const uint32_t L,
    std::vector<std::vector<uint8_t>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint8_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);


template void searchNaiveCAGRA<float>(const uint32_t k, const uint32_t L, uint32_t offset,
    std::vector<std::vector<float>>& data,
    std::vector<std::vector<uint32_t>>& index,
    std::vector<std::vector<float>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);
template void searchNaiveCAGRA<uint32_t>(const uint32_t k, const uint32_t L, uint32_t offset,
    std::vector<std::vector<uint32_t>>& data,
    std::vector<std::vector<uint32_t>>& index,
    std::vector<std::vector<uint32_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);
template void searchNaiveCAGRA<uint8_t>(const uint32_t k, const uint32_t L, uint32_t offset,
    std::vector<std::vector<uint8_t>>& data,
    std::vector<std::vector<uint32_t>>& index,
    std::vector<std::vector<uint8_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);


template void searchTranslatedCAGRA<float>(const uint32_t k, const uint32_t L,
    std::vector<std::vector<float>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint32_t>>& translation,
    std::vector<std::vector<float>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);
template void searchTranslatedCAGRA<uint32_t>(const uint32_t k, const uint32_t L,
    std::vector<std::vector<uint32_t>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint32_t>>& translation,
    std::vector<std::vector<uint32_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);
template void searchTranslatedCAGRA<uint8_t>(const uint32_t k, const uint32_t L,
    std::vector<std::vector<uint8_t>>& data,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint32_t>>& translation,
    std::vector<std::vector<uint8_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);



