#include <string>
#include <vector>
#include <iostream>

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
    uint32_t data_size = 0,
    uint32_t startLine = 0,
    uint32_t stopLine = 0);

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
    uint32_t data_size = 0,
    uint32_t startLine = 0,
    uint32_t stopLine = 0);

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
    uint32_t data_size = 0,
    uint32_t startLine = 0,
    uint32_t stopLine = 0);


template <typename T>
void searchNaiveCAGRA_disk(const uint32_t k, const uint32_t L, uint32_t offset,
    std::string data_file,
    std::vector<std::vector<uint32_t>>& index,
    std::vector<std::vector<T>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);

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
    long long* total_latency);

