#include <string>
#include <vector>

void merge(
    std::vector<std::vector<std::vector<uint32_t>>>& partitions, 
    std::vector<std::vector<uint32_t>>& merged);


void mergeShardAfterTranslation(omp_lock_t* locks, std::vector<std::vector<uint32_t>>& merged_index,
        std::vector<std::vector<uint32_t>>& index, std::vector<uint32_t>& idx_vec);


void mergeShardAfterTranslationGPU(omp_lock_t* locks, std::vector<std::vector<uint32_t>>& merged_index,
            std::vector<std::vector<uint32_t>>& index,
            std::vector<uint32_t>& idx_vec,
            uint32_t gpu_id);


void standardizeNeighborList(std::vector<std::vector<uint32_t>>& merged_index, uint32_t deg);