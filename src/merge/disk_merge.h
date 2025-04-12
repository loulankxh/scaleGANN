#include <string>

int DiskANN_merge(const std::string &vamana_prefix, const std::string &index_name, const std::string &idmaps_prefix,
    const uint64_t nshards, uint32_t max_degree,
    const std::string &output_vamana, const std::string &medoids_file, bool use_filters,
    const std::string &labels_to_medoids_file);

int scaleGANN_merge(const std::string base_folder,
    const uint64_t nshards, uint32_t max_degree, uint32_t constructed_deg,
    const std::string output_index_file,
    const std::string index_name);