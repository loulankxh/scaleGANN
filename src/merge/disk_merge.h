#include <string>

int scaleGANN_merge(const std::string base_folder,
    const uint64_t nshards, uint32_t max_degree, uint32_t constructed_deg,
    const std::string output_index_file,
    const std::string index_name);