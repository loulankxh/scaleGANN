#include <string>
#include <vector>

template <typename T>
void read(
        const std::string data_file, std::vector<std::vector<T>>& data,
        const std::string index_file, std::vector<std::vector<uint32_t>>& index,
        const std::string query_file, std::vector<std::vector<T>>& query,
        const std::string truth_file, std::vector<std::vector<uint32_t>>& groundTruth);

template <typename T>
void readExceptIndex(
        const std::string data_file, std::vector<std::vector<T>>& data,
        const std::string query_file, std::vector<std::vector<T>>& query,
        const std::string truth_file, std::vector<std::vector<uint32_t>>& groundTruth);


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
    uint32_t stopLine = 0);

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
    long long* total_latency);

template <typename T>
void searchNaiveCAGRA(const uint32_t k, const uint32_t L, uint32_t offset,
    std::vector<std::vector<T>>& data,
    std::vector<std::vector<uint32_t>>& index,
    std::vector<std::vector<T>>& query,
    std::vector<std::vector<uint32_t>>& result,
    std::vector<std::vector<float>>& distances,
    uint32_t* total_visited,
    uint32_t* total_distance_cmp,
    long long* total_latency);

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
    long long* total_latency);


double get_recall(uint32_t num_queries, uint32_t k,
        std::vector<std::vector<uint32_t>>& result,
        std::vector<std::vector<uint32_t>>& groundTruth);