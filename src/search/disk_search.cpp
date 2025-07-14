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

#include "search.hpp"
#include "disk_search.h"
#include "../utils/fileUtils.h"
#include "../utils/distance.h"
#include "../../DiskANN/include/neighbor.h"


template <typename T>
void search_disk(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::vector<std::vector<uint32_t>>& index,
    std::vector<std::vector<T>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t data_size,
    uint32_t startLine,
    uint32_t stopLine){

    if (k > L){
        throw std::invalid_argument("L should be no smaller than K.");
    }

    std::string file_type = "";
    uint32_t dotPos = data_file.find_last_of('.');
    if (dotPos == std::string::npos) {
        throw std::invalid_argument("File does not have a valid suffix.");
    }
    std::string suffix = data_file.substr(dotPos + 1);
    if(suffix.size() >= 4 && suffix.substr(suffix.size() - 4) == "vecs") file_type = "vecs";
    else if(suffix.size() >= 3 && suffix.substr(suffix.size() - 3) == "bin") file_type = "bin";
    else throw std::invalid_argument("File does not have a valid suffix.");

    std::ifstream data_reader(data_file, std::ios::binary);
    uint32_t npts = data_size;
    uint32_t dim;
    if(file_type == "bin") 
        data_reader.read(reinterpret_cast<char*>(&npts), sizeof(uint32_t));
    data_reader.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    uint32_t num_queries = query.size();
    result.resize(num_queries);
    distances.resize(num_queries);

    std::vector<uint32_t> count_visited(num_queries);
    std::vector<uint32_t> count_distances(num_queries);


    // #pragma omp parallel for schedule(static)
    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        auto startTime = std::chrono::high_resolution_clock::now();

        result[num_q].resize(k);
        distances[num_q].resize(k);
        std::vector<T> cur_query = query[num_q];
        diskann::NeighborPriorityQueue L_list(L);
        std::vector<diskann::Neighbor> expanded_nodes(0);
        std::unordered_map<uint32_t, bool> visited;


        uint32_t sid = startLine;
        uint32_t eid = npts;
        if(stopLine > 0){
            assert(data_size == stopLine - startLine);
            eid = stopLine;
        }
        std::vector<uint32_t> start(L);
        random_start_points(L, sid, eid, start);
        for (uint32_t id: start){
            size_t pos = 0;
            if(file_type == "bin")   
                pos = sizeof(uint32_t) * 2 + sizeof(T) * id * dim;
            else   
                pos = (sizeof(uint32_t) + sizeof(T) * dim) * id + sizeof(uint32_t);
            data_reader.seekg(pos);
            std::vector<T> node(dim);
            data_reader.read(reinterpret_cast<char*>(node.data()), sizeof(T) * dim);
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
                
                if(file_type == "vecs" && nb_id >= npts){ // GGNN Baseline build has bugs in Laion100M
                    continue;             
                }
                assert(nb_id < npts);

                if(visited.count(nb_id) == 0 || ((visited.count(nb_id) > 0) && (visited[nb_id]==false))){
                    visited[nb_id] = true;
                    count_visited[num_q] ++;

                    size_t pos = 0;
                    if(file_type == "bin")   
                        pos = sizeof(uint32_t) * 2 + sizeof(T) * nb_id * dim;
                    else   
                        pos = (sizeof(uint32_t) + sizeof(T) * dim) * nb_id + sizeof(uint32_t);
                    data_reader.seekg(pos);
                    std::vector<T> node(dim);
                    data_reader.read(reinterpret_cast<char*>(node.data()), sizeof(T) * dim);
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
void search_disk_scalegann(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::string index_file,
    std::vector<std::vector<T>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t data_size,
    uint32_t startLine,
    uint32_t stopLine){

    if (k > L){
        throw std::invalid_argument("L should be no smaller than K.");
    }

    std::string file_type = "";
    uint32_t dotPos = data_file.find_last_of('.');
    if (dotPos == std::string::npos) {
        throw std::invalid_argument("File does not have a valid suffix.");
    }
    std::string suffix = data_file.substr(dotPos + 1);
    if(suffix.size() >= 4 && suffix.substr(suffix.size() - 4) == "vecs") file_type = "vecs";
    else if(suffix.size() >= 3 && suffix.substr(suffix.size() - 3) == "bin") file_type = "bin";
    else throw std::invalid_argument("File does not have a valid suffix.");

    std::ifstream data_reader(data_file, std::ios::binary);
    uint32_t npts = data_size;
    uint32_t dim;
    if(file_type == "bin") 
        data_reader.read(reinterpret_cast<char*>(&npts), sizeof(uint32_t));
    data_reader.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    std::ifstream index_reader(index_file, std::ios::binary);
    if (!index_reader) {
        std::cerr << "Failed to open index file." << std::endl;
        return;
    }
    char dtype_string[4];
    index_reader.read(dtype_string, 4);
    int version = read_once<int>(index_reader);
    uint32_t rows = read_once<std::uint32_t>(index_reader);
    uint32_t idx_dim = read_once<std::uint32_t>(index_reader);
    uint32_t deg = read_once<std::uint32_t>(index_reader);
    unsigned short metric =  read_once<unsigned short>(index_reader);
    read_header(index_reader);
    size_t index_head = index_reader.tellg();

    uint32_t num_queries = query.size();
    result.resize(num_queries);
    distances.resize(num_queries);

    std::vector<uint32_t> count_visited(num_queries);
    std::vector<uint32_t> count_distances(num_queries);


    // #pragma omp parallel for schedule(static)
    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        auto startTime = std::chrono::high_resolution_clock::now();

        result[num_q].resize(k);
        distances[num_q].resize(k);
        std::vector<T> cur_query = query[num_q];
        diskann::NeighborPriorityQueue L_list(L);
        std::vector<diskann::Neighbor> expanded_nodes(0);
        std::unordered_map<uint32_t, bool> visited;


        uint32_t sid = startLine;
        uint32_t eid = npts;
        if(stopLine > 0){
            assert(data_size == stopLine - startLine);
            eid = stopLine;
        }
        std::vector<uint32_t> start(L);
        random_start_points(L, sid, eid, start);
        for (uint32_t id: start){
            size_t pos = 0;
            if(file_type == "bin")   
                pos = sizeof(uint32_t) * 2 + sizeof(T) * id * dim;
            else   
                pos = (sizeof(uint32_t) + sizeof(T) * dim) * id + sizeof(uint32_t);
            data_reader.seekg(pos);
            std::vector<T> node(dim);
            data_reader.read(reinterpret_cast<char*>(node.data()), sizeof(T) * dim);
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
            
            // std::vector<uint32_t> neighbors = index[n];
            size_t idx_pos = index_head + n * deg * sizeof(uint32_t);
            index_reader.seekg(idx_pos);
            std::vector<uint32_t> neighbors(deg);
            index_reader.read(reinterpret_cast<char*>(neighbors.data()), sizeof(uint32_t) * deg);
            // #pragma omp parallel for schedule(static)
            for (uint32_t iter = 0; iter< neighbors.size(); iter++){
                uint32_t nb_id = neighbors[iter];
                
                if(file_type == "vecs" && nb_id >= npts){ // GGNN Baseline build has bugs in Laion100M
                    continue;             
                }
                assert(nb_id < npts);

                if(visited.count(nb_id) == 0 || ((visited.count(nb_id) > 0) && (visited[nb_id]==false))){
                    visited[nb_id] = true;
                    count_visited[num_q] ++;

                    size_t pos = 0;
                    if(file_type == "bin")   
                        pos = sizeof(uint32_t) * 2 + sizeof(T) * nb_id * dim;
                    else   
                        pos = (sizeof(uint32_t) + sizeof(T) * dim) * nb_id + sizeof(uint32_t);
                    data_reader.seekg(pos);
                    std::vector<T> node(dim);
                    data_reader.read(reinterpret_cast<char*>(node.data()), sizeof(T) * dim);
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
void search_disk_diskann(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::string index_file,
    std::vector<std::vector<T>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t data_size,
    uint32_t startLine,
    uint32_t stopLine){

    if (k > L){
        throw std::invalid_argument("L should be no smaller than K.");
    }

    std::string file_type = "";
    uint32_t dotPos = data_file.find_last_of('.');
    if (dotPos == std::string::npos) {
        throw std::invalid_argument("File does not have a valid suffix.");
    }
    std::string suffix = data_file.substr(dotPos + 1);
    if(suffix.size() >= 4 && suffix.substr(suffix.size() - 4) == "vecs") file_type = "vecs";
    else if(suffix.size() >= 3 && suffix.substr(suffix.size() - 3) == "bin") file_type = "bin";
    else throw std::invalid_argument("File does not have a valid suffix.");

    std::ifstream data_reader(data_file, std::ios::binary);
    uint32_t npts = data_size;
    uint32_t dim;
    if(file_type == "bin") 
        data_reader.read(reinterpret_cast<char*>(&npts), sizeof(uint32_t));
    data_reader.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    std::ifstream index_reader(index_file.c_str(), std::ios::binary);
    std::vector<size_t> index_nnbrs(npts+1);
    size_t expected_file_size;
    index_reader.read((char *)&expected_file_size, sizeof(uint64_t));
    uint32_t input_width;
    index_reader.read((char *)&input_width, sizeof(uint32_t));
    uint64_t vamana_index_frozen = 0;
    index_reader.read((char *)&vamana_index_frozen, sizeof(uint64_t));
    uint32_t medoid;
    index_reader.read((char *)&medoid, sizeof(uint32_t));
    size_t read_data_size =
        sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t);
    size_t index_head = index_reader.tellg();
    uint32_t nnbrs = 0;
    index_nnbrs[0] = 0;
    uint32_t iter = 1;
    while (read_data_size < expected_file_size){
        index_reader.seekg(read_data_size);
        index_reader.read((char *)&nnbrs, sizeof(uint32_t));
        index_nnbrs[iter] = index_nnbrs[iter-1] + nnbrs;
        iter ++;
        read_data_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
    }


    uint32_t num_queries = query.size();
    result.resize(num_queries);
    distances.resize(num_queries);

    std::vector<uint32_t> count_visited(num_queries);
    std::vector<uint32_t> count_distances(num_queries);


    // #pragma omp parallel for schedule(static)
    for(uint32_t num_q = 0; num_q < num_queries; num_q++){
        auto startTime = std::chrono::high_resolution_clock::now();

        result[num_q].resize(k);
        distances[num_q].resize(k);
        std::vector<T> cur_query = query[num_q];
        diskann::NeighborPriorityQueue L_list(L);
        std::vector<diskann::Neighbor> expanded_nodes(0);
        std::unordered_map<uint32_t, bool> visited;


        uint32_t sid = startLine;
        uint32_t eid = npts;
        if(stopLine > 0){
            assert(data_size == stopLine - startLine);
            eid = stopLine;
        }
        std::vector<uint32_t> start(L);
        random_start_points(L, sid, eid, start);
        for (uint32_t id: start){
            size_t pos = 0;
            if(file_type == "bin")   
                pos = sizeof(uint32_t) * 2 + sizeof(T) * id * dim;
            else   
                pos = (sizeof(uint32_t) + sizeof(T) * dim) * id + sizeof(uint32_t);
            data_reader.seekg(pos);
            std::vector<T> node(dim);
            data_reader.read(reinterpret_cast<char*>(node.data()), sizeof(T) * dim);
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
            
            // std::vector<uint32_t> neighbors = index[n];
            size_t idx_pos = index_head + n * sizeof(uint32_t) + index_nnbrs[n] * sizeof(uint32_t);
            index_reader.seekg(idx_pos);
            uint32_t nnbrs = 0;
            index_reader.read((char *)&nnbrs, sizeof(uint32_t));
            std::vector<uint32_t> neighbors(nnbrs);
            index_reader.read(reinterpret_cast<char*>(neighbors.data()), sizeof(uint32_t) * nnbrs);
            // #pragma omp parallel for schedule(static)
            for (uint32_t iter = 0; iter< neighbors.size(); iter++){
                uint32_t nb_id = neighbors[iter];
                
                if(file_type == "vecs" && nb_id >= npts){ // GGNN Baseline build has bugs in Laion100M
                    continue;             
                }
                assert(nb_id < npts);

                if(visited.count(nb_id) == 0 || ((visited.count(nb_id) > 0) && (visited[nb_id]==false))){
                    visited[nb_id] = true;
                    count_visited[num_q] ++;

                    size_t pos = 0;
                    if(file_type == "bin")   
                        pos = sizeof(uint32_t) * 2 + sizeof(T) * nb_id * dim;
                    else   
                        pos = (sizeof(uint32_t) + sizeof(T) * dim) * nb_id + sizeof(uint32_t);
                    data_reader.seekg(pos);
                    std::vector<T> node(dim);
                    data_reader.read(reinterpret_cast<char*>(node.data()), sizeof(T) * dim);
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
void searchNaiveCAGRA_disk(const uint32_t k, const uint32_t L, uint32_t offset,
    std::string data_file,
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


    std::ifstream data_reader(data_file, std::ios::binary);
    uint32_t npts;
    uint32_t dim;
    data_reader.read(reinterpret_cast<char*>(&npts), sizeof(uint32_t));
    data_reader.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    uint32_t num_queries = query.size();
    result.resize(num_queries);
    distances.resize(num_queries);

    std::vector<uint32_t> count_visited(num_queries);
    std::vector<uint32_t> count_distances(num_queries);

    // #pragma omp parallel for schedule(static)
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
            size_t pos = sizeof(uint32_t) * 2 + sizeof(T) * (id + offset) * dim;
            data_reader.seekg(pos);
            std::vector<T> node(dim);
            data_reader.read(reinterpret_cast<char*>(node.data()), sizeof(T) * dim);
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

                    size_t pos = sizeof(uint32_t) * 2 + sizeof(T) * (nb_id + offset) * dim;
                    data_reader.seekg(pos);
                    std::vector<T> node(dim);
                    data_reader.read(reinterpret_cast<char*>(node.data()), sizeof(T) * dim);
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
void searchTranslatedCAGRA_disk(const uint32_t k, const uint32_t L,
    std::string data_file,
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


    std::ifstream data_reader(data_file, std::ios::binary);
    uint32_t npts;
    uint32_t dim;
    data_reader.read(reinterpret_cast<char*>(&npts), sizeof(uint32_t));
    data_reader.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    uint32_t num_queries = query.size();
    result.resize(num_queries);
    distances.resize(num_queries);

    std::vector<uint32_t> count_visited(num_queries);
    std::vector<uint32_t> count_distances(num_queries);

    // #pragma omp parallel for schedule(static)
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
            size_t pos = sizeof(uint32_t) * 2 + sizeof(T) * (translation[id][1]) * dim;
            data_reader.seekg(pos);
            std::vector<T> node(dim);
            data_reader.read(reinterpret_cast<char*>(node.data()), sizeof(T) * dim);
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

                    size_t pos = sizeof(uint32_t) * 2 + sizeof(T) * (translation[nb_id][1]) * dim;
                    data_reader.seekg(pos);
                    std::vector<T> node(dim);
                    data_reader.read(reinterpret_cast<char*>(node.data()), sizeof(T) * dim);
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




template void search_disk<float>(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<float>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t data_size,
    uint32_t startLine,
    uint32_t stopLine);
template void search_disk<uint32_t>(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint32_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t data_size,
    uint32_t startLine,
    uint32_t stopLine);
template void search_disk<uint8_t>(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint8_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t data_size,
    uint32_t startLine,
    uint32_t stopLine);


template void search_disk_scalegann<float>(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::string index_file,
    std::vector<std::vector<float>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t data_size,
    uint32_t startLine,
    uint32_t stopLine);
template void search_disk_scalegann<uint32_t>(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::string index_file,
    std::vector<std::vector<uint32_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t data_size,
    uint32_t startLine,
    uint32_t stopLine);
template void search_disk_scalegann<uint8_t>(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::string index_file,
    std::vector<std::vector<uint8_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t data_size,
    uint32_t startLine,
    uint32_t stopLine);


template void search_disk_diskann<float>(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::string index_file,
    std::vector<std::vector<float>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t data_size,
    uint32_t startLine,
    uint32_t stopLine);
template void search_disk_diskann<uint32_t>(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::string index_file,
    std::vector<std::vector<uint32_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t data_size,
    uint32_t startLine,
    uint32_t stopLine);
template void search_disk_diskann<uint8_t>(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::string index_file,
    std::vector<std::vector<uint8_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency,
    uint32_t data_size,
    uint32_t startLine,
    uint32_t stopLine);


    template void searchNaiveCAGRA_disk<float>(const uint32_t k, const uint32_t L, uint32_t offset,
    std::string data_file,
    std::vector<std::vector<uint32_t>>& index,
    std::vector<std::vector<float>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);
template void searchNaiveCAGRA_disk<uint32_t>(const uint32_t k, const uint32_t L, uint32_t offset,
    std::string data_file,
    std::vector<std::vector<uint32_t>>& index,
    std::vector<std::vector<uint32_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);
template void searchNaiveCAGRA_disk<uint8_t>(const uint32_t k, const uint32_t L, uint32_t offset,
    std::string data_file,
    std::vector<std::vector<uint32_t>>& index,
    std::vector<std::vector<uint8_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);


template void searchTranslatedCAGRA_disk<float>(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint32_t>>& translation,
    std::vector<std::vector<float>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);
template void searchTranslatedCAGRA_disk<uint32_t>(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint32_t>>& translation,
    std::vector<std::vector<uint32_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);
template void searchTranslatedCAGRA_disk<uint8_t>(const uint32_t k, const uint32_t L,
    std::string data_file,
    std::vector<std::vector<uint32_t>>& index, 
    std::vector<std::vector<uint32_t>>& translation,
    std::vector<std::vector<uint8_t>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);



