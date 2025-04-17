#include <string>
#include <vector>

template <typename T>
void diskANN_shard_data_into_clusters_with_ram_budget(const std::string data_file, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base, std::string prefix_path);

template <typename T>
void diskANN_partitions_with_ram_budget(const std::string data_file, double sampling_rate, double ram_budget,
    size_t graph_degree, const std::string prefix_path, size_t k_base);

template <typename T>
void scaleGANN_partitions_with_ram_budget(const std::string data_file, const double sampling_rate, double ram_budget,
    size_t graph_degree, const std::string prefix_path, size_t k_base, uint32_t num_parts = 0, float epsilon = 2, size_t max_k_means_reps = 15);