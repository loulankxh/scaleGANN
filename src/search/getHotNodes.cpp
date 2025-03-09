#include <string>
#include <vector>
#include <iostream>
#include <fstream>

template <typename T>
void read(
        const std::string data_file, std::vector<std::vector<T>>& data,
        const std::string index_file, std::vector<std::vector<uint32_t>>& index,
        const std::string query_file, std::vector<std::vector<T>>& query,
        const std::string truth_file, std::vector<std::vector<uint32_t>>& groundTruth){
    
    readFile<T>(data_file, data);
    readIndex(index_file, index);
    read_query<T>(query_file, query);
    read_groundTruth(truth_file, groundTruth);
}